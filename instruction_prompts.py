"""
Dictionary of Gemma3 instruction prompts which can be selected in args.
"""


DIVERSE_ORIGINAL = """This artificial life simulation was optimised to produce a simulation which sequentially follows the list PREVIOUS TARGET PROMPTS:
'{all_prompts}'.
The aim is to facilitate open-ended evolution of artificial life to discover new, interesting life forms - especially ones humans have never seen before.

You are in iteration {i} of the evolution, and your task is to provide the NEXT TARGET PROMPT for the next stage of the artificial life evolution, to follow on from the previous prompts and simulation. Your aim is to create a diverse, interesting and new life form - feel free to explore prompt space in unexpected and surprising ways. Be creative and be prepared to take risks! Your NEXT TARGET PROMPT should be macroscopically lifelike and meaningfully from the previous prompts in order to evolve open-ended life forms. Use your imagination, but keep your target prompt simple and concise in a FEW WORDS only. The algorithm will then append NEW TARGET PROMPT to the list of PREVIOUS TARGET PROMPTS and optimise the simulation parameters to create a simulation which matches this sequence of prompts.

ONLY output the new target prompt and nothing else. Keep it clear and concise. Have fun!

NEXT TARGET PROMPT: """

DIVERSE_NOVELTY_BONUS = """This artificial life simulation is evolving through a sequence of prompts:
'{all_prompts}'.  The goal is to discover *completely new* and unexpected life forms, beyond what humans have already imagined.

You are in iteration {i}.  Your task is to propose the NEXT TARGET PROMPT.  **Prioritize novelty and surprise.**  A prompt that is significantly different from previous prompts, *even if it seems strange*, is highly desirable.  Think about what hasn't been explored yet.  Focus on macroscopically lifelike features, but don't be afraid to push boundaries.  Keep the prompt concise (a few words).

ONLY output the new target prompt.

NEXT TARGET PROMPT: """


DIVERSE_CONSTRAINTS_AND_OPPORTUNITIES = """The artificial life simulation has followed these prompts:
'{all_prompts}'.  Consider these as *constraints* – things that have already been explored.

You are in iteration {i}.  Your task is to propose the NEXT TARGET PROMPT.  Think of this as identifying an *opportunity* for evolution – a direction that is significantly different from the past, but still potentially leads to a macroscopically lifelike outcome.  Be bold and creative!  A few words only.

ONLY output the new target prompt.

NEXT TARGET PROMPT: """


DIVERSE_FEW_SHOT = """The artificial life simulation has followed these prompts:
'{all_prompts}'.

Here are some examples of how to evolve the prompts, prioritizing diversity:
- Previous: 'floating jellyfish', Next: 'crystalline forests'
- Previous: 'burrowing worms', Next: 'aerial predators'
- Previous: 'bioluminescent fungi', Next: 'magnetic swarms'

You are in iteration {i}.  Your task is to propose the NEXT TARGET PROMPT, following this pattern of diverse evolution.  Be creative and concise.

ONLY output the new target prompt.

NEXT TARGET PROMPT: """

SHORT = """The simulation has evolved through: '{all_prompts}'.

Iteration {i}: Propose the NEXT TARGET PROMPT to drive further evolution.  Prioritize novelty and surprise.  Aim for macroscopically lifelike forms, but don't be afraid to be unexpected.  Keep it concise (a few words).

NEXT TARGET PROMPT: """

SHORT_BEHAVIOURS = """The simulation has evolved through: '{all_prompts}'.

Iteration {i}: Propose the NEXT TARGET PROMPT.  Focus on evolving life forms with interesting *movement patterns*, *social behaviors*, or *unique adaptations*.  Prioritize novelty.  Keep it concise.

NEXT TARGET PROMPT: """

DIVERSE_OPEN_ENDED = """This artificial life simulation has been optimised to follow this sequence of prompts:
'{all_prompts}'.
Consider these as constraints: ecological niches that have already been explored.

You are in iteration {i}.  Your task is to propose the NEXT TARGET PROMPT to determine the next stage of evolution.  This is an opportunity to propose a direction that is significantly different from the past, but leads to interesting lifelike behaviour.  Can we recreate open-ended evolution of life?  Be bold and creative!  ONLY output the new target prompt.

NEXT TARGET PROMPT: """

prompts = {
    "diverse_original": DIVERSE_ORIGINAL,
    "diverse_novelty_bonus": DIVERSE_NOVELTY_BONUS,
    "diverse_constraints_and_opportunities": DIVERSE_CONSTRAINTS_AND_OPPORTUNITIES,
    "diverse_few_shot": DIVERSE_FEW_SHOT,
    "short": SHORT,
    "short_behaviours": SHORT_BEHAVIOURS,
    "diverse_open_ended": DIVERSE_OPEN_ENDED
}