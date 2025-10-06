from collections import deque, defaultdict

import tqdm
import numpy as np
from loguru import logger

import jax
import jax.numpy as jnp
import optax
import rlax

from yahtzotron.game import play_tournament
from yahtzotron.interactive import print_score

REWARD_NORM = 100
WINNING_REWARD = 100

MINIMUM_LOGIT = jnp.finfo(jnp.float32).min


def entropy(logits):
    """Compute entropy from given logits."""
    probs = jax.nn.softmax(logits)
    logprobs = jax.nn.log_softmax(logits)
    return -jnp.sum(probs * logprobs, axis=-1)


def cross_entropy(logits, actions):
    """Compute cross-entropy from given logits and labels."""
    logprob = jax.nn.log_softmax(logits)
    labels = jax.nn.one_hot(actions, logits.shape[-1])
    return -jnp.sum(labels * logprob, axis=1)


def compile_loss_function(type_, network):
    """Compile loss function to use during training.

    type_ can be either "a2c" (using a policy loss for the actor)
    or "supervised" (using cross-entropy).
    """

    def loss(
        weights,
        observations,
        actions,
        rewards,
        td_lambda=0.2,
        discount=0.99,
        policy_cost=0.25,
        entropy_cost=1e-3,
    ):
        """Actor-critic loss."""
        logits, values = network(weights, observations)
        values = jnp.append(values, jnp.sum(rewards))

        # replace -inf values by tiny finite value
        logits = jnp.maximum(logits, MINIMUM_LOGIT)

        td_errors = rlax.td_lambda(
            v_tm1=values[:-1],
            r_t=rewards,
            discount_t=jnp.full_like(rewards, discount),
            v_t=values[1:],
            lambda_=td_lambda,
        )
        critic_loss = jnp.mean(td_errors ** 2)

        if type_ == "a2c":
            actor_loss = rlax.policy_gradient_loss(
                logits_t=logits,
                a_t=actions,
                adv_t=td_errors,
                w_t=jnp.ones(td_errors.shape[0]),
            )
        elif type_ == "supervised":
            actor_loss = jnp.mean(cross_entropy(logits, actions))

        entropy_loss = -jnp.mean(entropy(logits))

        return policy_cost * actor_loss, critic_loss, entropy_cost * entropy_loss

    return jax.jit(loss)


def compile_vectorized_loss_function(type_, network):
    """Compile vectorized loss function for batched training across multiple trajectories.
    
    This version processes multiple complete trajectories in parallel for better GPU utilization.
    """
    
    def vectorized_loss(
        weights,
        batch_observations,  # Shape: [batch_size, seq_len, obs_dim]
        batch_actions,       # Shape: [batch_size, seq_len]
        batch_rewards,       # Shape: [batch_size, seq_len]
        td_lambda=0.2,
        discount=0.99,
        policy_cost=0.25,
        entropy_cost=1e-3,
    ):
        """Vectorized actor-critic loss across multiple trajectories."""
        batch_size, seq_len = batch_observations.shape[:2]
        
        # Reshape for vectorized network forward pass
        flat_observations = batch_observations.reshape(-1, batch_observations.shape[-1])
        flat_logits, flat_values = network(weights, flat_observations)
        
        # Reshape back to batch format
        logits = flat_logits.reshape(batch_size, seq_len, -1)
        values = flat_values.reshape(batch_size, seq_len)
        
        # Add terminal values (sum of rewards for each trajectory)
        terminal_values = jnp.sum(batch_rewards, axis=1, keepdims=True)
        values = jnp.concatenate([values, terminal_values], axis=1)
        
        # Replace -inf values by tiny finite value
        logits = jnp.maximum(logits, MINIMUM_LOGIT)
        
        # Vectorized TD-lambda calculation for each trajectory
        def single_trajectory_td(rewards, vals):
            return rlax.td_lambda(
                v_tm1=vals[:-1],
                r_t=rewards,
                discount_t=jnp.full_like(rewards, discount),
                v_t=vals[1:],
                lambda_=td_lambda,
            )
        
        # Apply TD-lambda to each trajectory in the batch
        td_errors = jax.vmap(single_trajectory_td)(batch_rewards, values)
        
        # Flatten for loss calculations
        flat_td_errors = td_errors.reshape(-1)
        flat_actions = batch_actions.reshape(-1)
        flat_logits = logits.reshape(-1, logits.shape[-1])
        
        critic_loss = jnp.mean(flat_td_errors ** 2)
        
        if type_ == "a2c":
            actor_loss = rlax.policy_gradient_loss(
                logits_t=flat_logits,
                a_t=flat_actions,
                adv_t=flat_td_errors,
                w_t=jnp.ones(flat_td_errors.shape[0]),
            )
        elif type_ == "supervised":
            actor_loss = jnp.mean(cross_entropy(flat_logits, flat_actions))
        
        entropy_loss = -jnp.mean(entropy(flat_logits))
        
        return policy_cost * actor_loss, critic_loss, entropy_cost * entropy_loss
    
    return jax.jit(vectorized_loss)


def compile_sgd_step(loss_func, optimizer):
    def sgd_step(weights, opt_state, observations, actions, rewards, **loss_kwargs):
        """Does a step of SGD over a trajectory."""
        total_loss = lambda *args: sum(loss_func(*args, **loss_kwargs))
        gradients = jax.grad(total_loss)(weights, observations, actions, rewards)
        updates, opt_state = optimizer.update(gradients, opt_state)
        weights = optax.apply_updates(weights, updates)
        return weights, opt_state

    return jax.jit(sgd_step)


def compile_batch_sgd_step(loss_func, optimizer):
    """Compile a batched SGD step that processes multiple trajectories efficiently."""
    
    def single_trajectory_loss_and_grad(weights, obs, acts, rews, entropy_cost, td_lambda):
        """Compute loss and gradients for a single trajectory."""
        total_loss = lambda *args: sum(loss_func(*args, td_lambda=td_lambda, entropy_cost=entropy_cost))
        loss_val = total_loss(weights, obs, acts, rews)
        gradients = jax.grad(total_loss)(weights, obs, acts, rews)
        return loss_val, gradients
    
    # Vectorize across multiple trajectories
    batched_loss_and_grad = jax.vmap(
        single_trajectory_loss_and_grad, 
        in_axes=(None, 0, 0, 0, None, None),  # weights, obs, acts, rews, entropy_cost, td_lambda
        out_axes=(0, 0)
    )
    
    def batch_sgd_step(weights, opt_state, batch_obs, batch_acts, batch_rews, **loss_kwargs):
        """Does a step of SGD over multiple trajectories efficiently."""
        # Extract scalar kwargs
        entropy_cost = loss_kwargs.get('entropy_cost', 1e-3)
        td_lambda = loss_kwargs.get('td_lambda', 0.2)
        
        # Compute loss and gradients for all trajectories in parallel
        losses, gradients = batched_loss_and_grad(weights, batch_obs, batch_acts, batch_rews, entropy_cost, td_lambda)
        
        # Average gradients across trajectories
        avg_gradients = jax.tree_map(lambda x: jnp.mean(x, axis=0), gradients)
        
        # Apply averaged gradients
        updates, opt_state = optimizer.update(avg_gradients, opt_state)
        weights = optax.apply_updates(weights, updates)
        
        # Return average loss for logging
        avg_loss = jnp.mean(losses)
        return weights, opt_state, avg_loss
    
    return jax.jit(batch_sgd_step)


def compile_vectorized_sgd_step(loss_func, optimizer):
    def vectorized_sgd_step(weights, opt_state, batch_observations, batch_actions, batch_rewards, **loss_kwargs):
        """Does a step of SGD over multiple trajectories in parallel."""
        total_loss = lambda *args: sum(loss_func(*args, **loss_kwargs))
        gradients = jax.grad(total_loss)(weights, batch_observations, batch_actions, batch_rewards)
        updates, opt_state = optimizer.update(gradients, opt_state)
        weights = optax.apply_updates(weights, updates)
        return weights, opt_state

    return jax.jit(vectorized_sgd_step)


def get_default_schedules(pretraining=False):
    """Get schedules for learning rate, entropy, TDlambda."""
    if pretraining:
        return dict(
            learning_rate=optax.constant_schedule(5e-3),
            entropy=optax.constant_schedule(1e-3),
            td_lambda=optax.constant_schedule(0.2),
        )

    return dict(
        learning_rate=optax.exponential_decay(1e-3, 60_000, decay_rate=0.2),
        entropy=(
            lambda count: 1e-3 * 0.1 ** (count / 80_000) if count < 80_000 else -1e-2
        ),
        td_lambda=optax.polynomial_schedule(0.2, 0.8, power=1, transition_steps=60_000),
    )


def train_a2c(
    base_agent,
    num_epochs,
    checkpoint_path=None,
    players_per_game=4,
    lr_schedule=None,
    entropy_schedule=None,
    td_lambda_schedule=None,
    pretraining=False,
    batch_size_multiplier=16,  # Increased default for better GPU utilization
):
    """Train advantage actor-critic (A2C) agent through self-play"""
    objective = base_agent._objective

    default_schedules = get_default_schedules(pretraining=pretraining)

    if lr_schedule is None:
        lr_schedule = default_schedules["learning_rate"]

    if entropy_schedule is None:
        entropy_schedule = default_schedules["entropy"]

    if td_lambda_schedule is None:
        td_lambda_schedule = default_schedules["td_lambda"]

    optimizer = optax.MultiSteps(
        optax.chain(optax.adam(learning_rate=1), optax.scale_by_schedule(lr_schedule)),
        players_per_game,
    )
    opt_state = optimizer.init(base_agent.get_weights())

    running_stats = defaultdict(lambda: deque(maxlen=1000))
    progress = tqdm.tqdm(range(num_epochs), dynamic_ncols=True)

    loss_type = "supervised" if pretraining else "a2c"
    loss_fn = compile_loss_function(loss_type, base_agent._network)
    
    # Use batched SGD for better GPU utilization when batch_size_multiplier > 1
    if batch_size_multiplier > 1:
        batch_sgd_step = compile_batch_sgd_step(loss_fn, optimizer)
        logger.info("Using GPU-optimized batched training with batch_size_multiplier={}", batch_size_multiplier)
        use_batched = True
    else:
        sgd_step = compile_sgd_step(loss_fn, optimizer)
        logger.info("Using standard training")
        use_batched = False

    best_score = -float("inf")

    if pretraining:
        greedy_agent = base_agent.clone()
        greedy_agent._be_greedy = True
        agents = [greedy_agent] * players_per_game
    else:
        agents = [base_agent] * players_per_game

    # Buffer to collect multiple trajectories before training
    trajectory_buffer = []
    score_buffer = []

    for i in progress:
        scores, trajectories = play_tournament(agents, record_trajectories=True)

        final_scores = [s.total_score() for s in scores]
        winner = np.argmax(final_scores)
        logger.info(
            "Player {} won with a score of {} (median {})",
            winner,
            final_scores[winner],
            np.median(final_scores),
        )
        logger.info(
            " Winning scorecard:\n{}",
            print_score(scores[winner]),
        )

        # Add current trajectories to buffer
        trajectory_buffer.extend(trajectories)
        score_buffer.extend(scores)

        # Debug output to show batching progress
        logger.debug(" Buffer size: {}/{} trajectories", len(trajectory_buffer), batch_size_multiplier * players_per_game)

        # Train when we have enough trajectories for efficient batching
        if len(trajectory_buffer) >= batch_size_multiplier * players_per_game:
            logger.info(" Training with batch of {} trajectories ({} observations)", 
                       len(trajectory_buffer), len(trajectory_buffer) * 36)
            
            weights = base_agent._weights
            loss_kwargs = dict(
                entropy_cost=entropy_schedule(i), td_lambda=td_lambda_schedule(i)
            )

            if use_batched:
                # Prepare batch of trajectories for parallel processing
                batch_observations = []
                batch_actions = []
                batch_rewards = []
                
                for p, (trajectory, score) in enumerate(zip(trajectory_buffer, score_buffer)):
                    observations, actions, rewards = zip(*trajectory)
                    assert sum(rewards) == score.total_score()

                    observations = np.stack(observations, axis=0)
                    actions = np.array(actions, dtype=np.int32)
                    rewards = np.array(rewards, dtype=np.float32) / REWARD_NORM

                    # Apply winning bonus if applicable
                    if objective == "win" and p == winner:
                        rewards[-1] += WINNING_REWARD / REWARD_NORM
                        

                    batch_observations.append(observations)
                    batch_actions.append(actions)
                    batch_rewards.append(rewards)
                
                # Convert to JAX arrays in batch format
                batch_observations = jnp.array(np.stack(batch_observations, axis=0))
                batch_actions = jnp.array(np.stack(batch_actions, axis=0))
                batch_rewards = jnp.array(np.stack(batch_rewards, axis=0))
                
                logger.debug(" Batch shapes: obs={}, acts={}, rews={}", 
                           batch_observations.shape, batch_actions.shape, batch_rewards.shape)
                
                # Single batched training step
                weights, opt_state, avg_loss = batch_sgd_step(
                    weights, opt_state, batch_observations, batch_actions, batch_rewards, **loss_kwargs
                )
                
                # Compute detailed loss components for logging (using first trajectory as representative)
                loss_components = loss_fn(
                    weights, batch_observations[0], batch_actions[0], batch_rewards[0], **loss_kwargs
                )
                avg_loss_components = [float(k) for k in loss_components]
                
            else:
                # Process each trajectory individually (original approach)
                all_loss_components = []
                
                for p, (trajectory, score) in enumerate(zip(trajectory_buffer, score_buffer)):
                    observations, actions, rewards = zip(*trajectory)
                    assert sum(rewards) == score.total_score()

                    # Convert to JAX arrays directly for better GPU performance
                    observations = jnp.array(np.stack(observations, axis=0))
                    actions = jnp.array(np.array(actions, dtype=np.int32))
                    rewards = jnp.array(np.array(rewards, dtype=np.float32) / REWARD_NORM)

                    # Apply winning bonus if applicable
                    if objective == "win" and p == winner:
                        rewards = rewards.at[-1].add(WINNING_REWARD / REWARD_NORM)

                    # Train on this trajectory
                    weights, opt_state = sgd_step(
                        weights, opt_state, observations, actions, rewards, **loss_kwargs
                    )

                    # Compute loss for logging
                    loss_components = loss_fn(
                        weights, observations, actions, rewards, **loss_kwargs
                    )
                    all_loss_components.append([float(k) for k in loss_components])

                # Average loss components for logging
                avg_loss_components = np.mean(all_loss_components, axis=0)

            # Log stats for each player
            for p, score in enumerate(score_buffer):
                epoch_stats = dict(
                    actor_loss=avg_loss_components[0],
                    critic_loss=avg_loss_components[1],
                    entropy_loss=avg_loss_components[2],
                    loss=sum(avg_loss_components),
                    score=score.total_score(),
                )
                for key, val in epoch_stats.items():
                    buf = running_stats[key]
                    if len(buf) == buf.maxlen:
                        buf.popleft()
                    buf.append(val)

            base_agent.set_weights(weights)

            if pretraining:
                greedy_agent.set_weights(weights)

            # Clear buffers after training
            trajectory_buffer.clear()
            score_buffer.clear()

        if i % 10 == 0:
            avg_score = np.mean(running_stats["score"])
            if avg_score > best_score + 1 and i > running_stats["score"].maxlen:
                best_score = avg_score

                if checkpoint_path is not None:
                    logger.warning(
                        " Saving checkpoint for average score {:.2f}", avg_score
                    )
                    base_agent.save(checkpoint_path)

            progress.set_postfix(
                {key: np.mean(val) for key, val in running_stats.items()}
            )

    # Train any remaining trajectories in the buffer at the end
    if len(trajectory_buffer) > 0:
        weights = base_agent._weights
        loss_kwargs = dict(
            entropy_cost=entropy_schedule(num_epochs-1), td_lambda=td_lambda_schedule(num_epochs-1)
        )

        # Process remaining trajectories
        all_observations = []
        all_actions = []
        all_rewards = []

        for p, (trajectory, score) in enumerate(zip(trajectory_buffer, score_buffer)):
            observations, actions, rewards = zip(*trajectory)
            assert sum(rewards) == score.total_score()

            observations = np.stack(observations, axis=0)
            actions = np.array(actions, dtype=np.int32)
            rewards = np.array(rewards, dtype=np.float32) / REWARD_NORM

            all_observations.append(observations)
            all_actions.append(actions)
            all_rewards.append(rewards)

        # Concatenate remaining trajectories
        batch_observations = np.concatenate(all_observations, axis=0)
        batch_actions = np.concatenate(all_actions, axis=0)
        batch_rewards = np.concatenate(all_rewards, axis=0)

        logger.debug(" Final batch size: {} observations", batch_observations.shape[0])

        # Final training step
        weights, opt_state = sgd_step(
            weights, opt_state, batch_observations, batch_actions, batch_rewards, **loss_kwargs
        )

        base_agent.set_weights(weights)
        if pretraining:
            greedy_agent.set_weights(weights)

    return base_agent


def train_strategy(agent, num_epochs, players_per_game=4, learning_rate=1e-3):
    """Train strategy net through supervised learning on observed final actions."""
    optimizer = optax.adam(learning_rate=learning_rate)
    opt_state = optimizer.init(agent.get_weights(strategy=True))

    num_rolls = 3

    @jax.jit
    def loss_fn(weights, observation, action, *args):
        loss = jnp.zeros(1)

        final_actions = action[num_rolls - 1 :: num_rolls]

        for i_roll in range(num_rolls - 1):
            logits = agent._strategy_network(weights, observation[i_roll::num_rolls])
            loss = loss + jnp.mean(cross_entropy(logits, final_actions))

        return loss

    sgd_step = compile_sgd_step(loss_fn, optimizer)
    running_loss = deque(maxlen=1000)

    progress = tqdm.tqdm(range(num_epochs), dynamic_ncols=True)

    for i in progress:
        _, trajectories = play_tournament(
            [agent] * players_per_game, record_trajectories=True
        )

        weights = agent.get_weights(strategy=True)

        for p in range(players_per_game):
            observations, actions, _ = zip(*trajectories[p])

            observations = np.stack(observations, axis=0)
            actions = np.array(actions, dtype=np.int32)

            rolls_left = observations[..., 0]
            for k in range(num_rolls):
                assert np.all(rolls_left[k::num_rolls] == num_rolls - k - 1)

            logger.debug(" observations {}: {}", p, observations)
            logger.debug(" actions {}: {}", p, actions)

            weights, opt_state = sgd_step(
                weights,
                opt_state,
                observations,
                actions,
                None,
            )

            loss = float(loss_fn(weights, observations, actions).item())

            if len(running_loss) == running_loss.maxlen:
                running_loss.popleft()

            running_loss.append(loss)

        agent.set_weights(weights, strategy=True)

        if i % 10 == 0:
            progress.set_postfix(dict(loss=np.mean(running_loss)))

    return agent
