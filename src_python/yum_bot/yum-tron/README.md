# Yum-tron

> State your prime directive! - "... to ... roll ..." 🤖 🎲

Yum-tron is a bot for [Yum](https://en.wikipedia.org/wiki/Yahtzee) (a Yahtzee variant), trained via advantage actor-critic (A2C) through self-play. Yum-tron is implemented through the JAX library ecosystem ([JAX](https://github.com/google/jax) + [Haiku](https://github.com/deepmind/dm-haiku) + [optax](https://github.com/deepmind/optax) + [rlax](https://github.com/deepmind/rlax)).

Yum is a game of chance played with 5 dice and involves making strategic decisions based on the outcome of your rolls early in the game. This makes for a surprisingly challenging task for reinforcement learning.

The bot is trained to play the Yum variant with 12 scoring categories including the special "Yum" category for 5-of-a-kind.

## Usage

Just clone the repository and run

```bash
$ pip install .
```

Then, you can use the Yum-tron command-line interface:

```
$ yum-tron --help
Usage: yum-tron [OPTIONS] COMMAND [ARGS]...

  This is Yum-tron, the friendly robot that beats you in Yum (Yahtzee variant).

Options:
  --version                       Show the version and exit.
  -v, --loglevel [debug|info|warning|error]
  --help                          Show this message and exit.

Commands:
  evaluate  Evaluate performance of trained agents.
  origin    Show Yahtzotron's origin story.
  play      Play a game against Yahtzotron.
  train     Train a new model through self-play.
```

Why don't you try a game against one of the pre-trained agents?

```bash
$ yum-tron play pretrained/yahtzee-score.pkl
```

#### Bonus

When you play Yum-tron, it is going to tell you what its current strategy is before every action (to teach us puny humans how to play):

```
> My turn!
> Roll #1: [3, 3, 3, 5, 6].
> I think I should go for Threes, so I'm keeping [3, 3, 3].
> Roll #2: [3, 3, 3, 3, 4].
> I think I should go for Threes or Yum, so I'm keeping [3, 3, 3, 3].
> Roll #3: [1, 3, 3, 3, 3].
> I'll pick the "Threes" category for that.
```
