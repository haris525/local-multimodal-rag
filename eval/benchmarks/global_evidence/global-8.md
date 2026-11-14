# global-8

Question: The agent-architectures survey presents five single-agent methods. Name them and, for each one, give the limitations the survey says it has.

Gold answer: ReAct, RAISE, Reflexion, AutoGPT+P and LATS (Section 3). ReAct: the model can repetitively generate the same thoughts and actions and fail to create new thoughts that finish the task and exit the ReAct loop (human feedback would help). RAISE: it struggles to understand complex logic, and its agents often hallucinate about their roles or knowledge (a sales agent may start writing Python code); fine-tuning helped, but hallucination remains a limitation. Reflexion: it is susceptible to non-optimal local minima solutions, and it keeps long-term memory in a sliding window rather than a database, so memory is limited by the model's token limit; it still has room to improve on tasks that need much diversity, exploration and reasoning. AutoGPT+P: tool selection accuracy varies (tools called inappropriately or stuck in loops), exploration can be illogical (looking for objects in the wrong place), and human interaction is limited (the agent cannot seek clarification and the user cannot modify or terminate the plan during execution). LATS: it uses more computational resources and takes more time than other single-agent methods, and it was tested only on relatively simple question answering benchmarks, not on scenarios with tool calling or complex reasoning.

Documents: ai-agent-architectures-survey-2024.pdf

## Evidence

- the five methods (Section 3.1)
  - ai-agent-architectures-survey-2024.pdf, page 4
  - "In this section, we highlight some notable single agent methods such as ReAct, RAISE, Reflexion, AutoGPT + P, and LATS."

- ReAct: repeats the same thoughts and actions
  - ai-agent-architectures-survey-2024.pdf, page 4
  - "the model can repetitively generate the same thoughts and actions and fail to create new thoughts to provoke finishing the task and exiting the ReAct loop."

- RAISE: complex logic
  - ai-agent-architectures-survey-2024.pdf, page 4
  - "First, RAISE struggles to understand complex logic, limiting its usefulness in many scenarios."

- RAISE: hallucination about roles
  - ai-agent-architectures-survey-2024.pdf, page 4
  - "Additionally, RAISE agents often hallucinated with respect to their roles or knowledge."

- Reflexion: local minima (the PDF has curly quotes, written here as ASCII)
  - ai-agent-architectures-survey-2024.pdf, page 5
  - "Primarily, Reflexion is susceptible to "non-optimal local minima solutions"."

- Reflexion: sliding-window long-term memory
  - ai-agent-architectures-survey-2024.pdf, page 5
  - "It also uses a sliding window for long-term memory, rather than a database. This means that the volume of long-term memory is limited by the token limit of the language model."

- AutoGPT+P: tool selection
  - ai-agent-architectures-survey-2024.pdf, page 6
  - "Accuracy of tool selection varies, with certain tools being called inappropriately or getting stuck in loops."

- AutoGPT+P: limited human interaction
  - ai-agent-architectures-survey-2024.pdf, page 6
  - "The framework also is limited in terms of human interaction, with the agent being unable to seek clarification and the user being unable to modify or terminate the plan during execution."

- LATS: compute and time
  - ai-agent-architectures-survey-2024.pdf, page 6
  - "LATS often uses more computational resources and takes more time to complete than other single-agent methods [36]."

- LATS: simple benchmarks
  - ai-agent-architectures-survey-2024.pdf, page 6
  - "The paper also uses relatively simple question answering benchmarks and has not been tested on more robust scenarios that involve involving tool calling or complex reasoning."

## Arithmetic and notes

The survey's Section 3.3 has exactly five method subsections (ReAct, RAISE, Reflexion, AUTOGPT + P, LATS), matching the Section 3.1 overview. Section 4 is about multi-agent methods and is out of scope. The "involve involving" typo is in the source.

## What makes it global

The five limitation paragraphs are on three pages (4, 5, 6), one per method subsection, separated by figures. A search for limitations returns two or three of them; listing all five needs the whole section.
