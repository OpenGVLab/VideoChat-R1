---
license: mit
extra_gated_prompt: >-
  You agree to not use the dataset to conduct experiments that cause harm to
  human subjects. Please note that the data in this dataset may be subject to
  other agreements. Before using the data, be sure to read the relevant
  agreements carefully to ensure compliant use. Video copyrights belong to the
  original video creators or platforms and are for academic research use only.
task_categories:
- visual-question-answering
- video-classification
extra_gated_fields:
  Name: text
  Company/Organization: text
  Country: text
  E-Mail: text
modalities:
- Video
- Text
configs:
- config_name: action_sequence
  data_files: json/action_sequence.json
- config_name: moving_count
  data_files: json/moving_count.json
- config_name: action_prediction
  data_files: json/action_prediction.json
- config_name: episodic_reasoning
  data_files: json/episodic_reasoning.json
- config_name: action_antonym
  data_files: json/action_antonym.json
- config_name: action_count
  data_files: json/action_count.json
- config_name: scene_transition
  data_files: json/scene_transition.json
- config_name: object_shuffle
  data_files: json/object_shuffle.json
- config_name: object_existence
  data_files: json/object_existence.json
- config_name: fine_grained_pose
  data_files: json/fine_grained_pose.json
- config_name: unexpected_action
  data_files: json/unexpected_action.json
- config_name: moving_direction
  data_files: json/moving_direction.json
- config_name: state_change
  data_files: json/state_change.json
- config_name: object_interaction
  data_files: json/object_interaction.json
- config_name: character_order
  data_files: json/character_order.json
- config_name: action_localization
  data_files: json/action_localization.json
- config_name: counterfactual_inference
  data_files: json/counterfactual_inference.json
- config_name: fine_grained_action
  data_files: json/fine_grained_action.json
- config_name: moving_attribute
  data_files: json/moving_attribute.json
- config_name: egocentric_navigation
  data_files: json/egocentric_navigation.json
language:
- en
size_categories:
- 1K<n<10K
---