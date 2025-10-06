from .yatzy import yatzy_rules
from .yahtzee import yahtzee_rules
from .yum import yum_rules

AVAILABLE_RULESETS = {r.name: r for r in (yatzy_rules, yahtzee_rules, yum_rules)}
