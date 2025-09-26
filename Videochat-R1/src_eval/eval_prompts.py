CLS_QA_TEMPLATE = """[PROBLEM]

Output your thought process within the <think> </think> tags, including analysis with either specific timestamps (xx.xx) or time ranges (xx.xx to xx.xx) in <timestep> </timestep> tags.

Then, provide your final answer within the <answer> </answer> tags.
"""

GROUND_TEMPLATE_THINK = """To accurately pinpoint the event "[EVENT]" in the video, determine the precise time period of the event.

Output your thought process within the <think> </think> tags, including analysis with either specific timestamps (xx.xx) or time ranges (xx.xx to xx.xx) in <timestep> </timestep> tags.

Then, provide the start and end times (in seconds, precise to two decimal places) in the format "start time to end time" within the <answer> </answer> tags. For example: "12.54 to 17.83"."""

GROUND_TEMPLATE_NOTHINK = """To accurately pinpoint the event "[EVENT]" in the video, determine the precise time period of the event.

Provide the start and end times (in seconds, precise to two decimal places) in the format "start time to end time" within the <answer> </answer> tags. For example: "12.54 to 17.83"."""

GQA_ANSWER = """Answer the question: "[QUESTION]" according to the content of the video. Select the answer from :[OPTION]. Provide your answer within the <answer> </answer> tags, output the corresponding letter of the option.
"""

GQA_THINK_GLUE = """Answer the question: "[QUESTION]" according to the content of the video. Select the answer from :[OPTION].

Output your thought process within the <think> </think> tags, including analysis with either specific timestamps (xx.xx) or time ranges (xx.xx to xx.xx) in <timestep> </timestep> tags.

Then, provide your answer within the <answer> </answer> tags, output the corresponding letter of the option.
"""

GQA_THINK_ANSWER_GLUE = """Answer the question: "[QUESTION]" according to the content of the video. Select the answer from :[OPTION].

Output your thought process within the <think> </think> tags, including analysis with either specific timestamps (xx.xx) or time ranges (xx.xx to xx.xx) in <timestep> </timestep> tags.

Then, provide your answer within the <answer> </answer> tags, output the corresponding letter of the option. At the same time, in the <glue> </glue> tags, present the precise time period in seconds of the video clips on which you base your answer to this question in the format of [(s1, e1), (s2, e2), ...]. For example: <answer>A</answer><glue>[(5.2, 10.4)]</glue>.
"""

GQA_ANSWER_GLUE = """Answer the question: "[QUESTION]" according to the content of the video. Select the answer from :[OPTION]. Provide your answer within the <answer> </answer> tags, output the corresponding letter of the option. At the same time, in the <glue> </glue> tags, present the precise time period in seconds of the video clips on which you base your answer to this question in the format of [(s1, e1), (s2, e2), ...]. For example: <answer>A</answer><glue>[(5.2, 10.4)]</glue>.
"""

QA_THINK = """Answer the question: "[QUESTION]" according to the content of the video. Select the answer from :[OPTION].

Output your thought process within the <think> </think> tags, including analysis with either specific timestamps (xx.xx) or time ranges (xx.xx to xx.xx) in <timestep> </timestep> tags.

Then, provide your answer within the <answer> </answer> tags, output the corresponding letter of the option.
"""


QA_NOTHINK = """Answer the question: "[QUESTION]" according to the content of the video. Select the answer from :[OPTION].

Provide your answer within the <answer> </answer> tags, output the corresponding letter of the option.
"""

TRACK_THINK = """Track the "[OBJECT]" in the video based on its initial coordinates "[START]". The output should be a list containing eight sublists. Each sublist includes four normalized coordinates [x0, y0, x1, y1] representing the bounding box of the object at specific time intervals.

Output your thought process within the <think> </think> tags.

Provide your answer within the <answer> </answer> tags as a list of eight sublists, where each sublist contains the normalized coordinates [x0, y0, x1, y1]. For example: <answer>[[0.1, 0.5, 0.3, 0.55], [0.72, 0.25, 0.84, 0.43], ...]</answer>.
"""

TRACK_NO_THINK = """Track the "[OBJECT]" in the video based on its initial coordinates "[START]". The output should be a list containing eight sublists. Each sublist includes four normalized coordinates [x0, y0, x1, y1] representing the bounding box of the object at specific time intervals.

Provide your answer within the <answer> </answer> tags as a list of eight sublists, where each sublist contains the normalized coordinates [x0, y0, x1, y1]. For example: <answer>[[0.1, 0.5, 0.3, 0.55], [0.72, 0.25, 0.84, 0.43], ...]</answer>.
"""