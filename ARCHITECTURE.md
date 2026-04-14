python_mini_metro/
|- .cursor/
|  |- rules/
|  |  |- run-tests-after-changes.mdc
|  |  |- update-architecture.mdc
|  |  |- update-game-rules.mdc
|  |  \- update-progress.mdc
|- .github/
|  \- workflows/
|     \- test.yml
|- src/
|  |- agent_play.py
|  |- config.py
|  |- env.py
|  |- gnn_lite_policy.py
|  |- jongno_env.py
|  |- jongno_policy.py
|  |- main.py
|  |- mediator.py
|  |- compare_jongno_policies.py
|  |- task_catalog.py
|  |- task_runner.py
|  |- train_jongno_gnn_lite.py
|  |- train_jongno_policy.py
|  |- travel_plan.py
|  |- type.py
|  |- utils.py
|  \- visualize_jongno_policy.py
|  |- entity/
|  |  |- get_entity.py
|  |  |- holder.py
|  |  |- metro.py
|  |  |- padding_segment.py
|  |  |- passenger.py
|  |  |- path.py
|  |  |- path_segment.py
|  |  |- segment.py
|  |  \- station.py
|  |- event/
|  |  |- convert.py
|  |  |- event.py
|  |  |- keyboard.py
|  |  |- mouse.py
|  |  \- type.py
|  |- geometry/
|  |  |- circle.py
|  |  |- cross.py
|  |  |- diamond.py
|  |  |- line.py
|  |  |- pentagon.py
|  |  |- point.py
|  |  |- polygon.py
|  |  |- rect.py
|  |  |- shape.py
|  |  |- star.py
|  |  |- triangle.py
|  |  |- type.py
|  |  \- utils.py
|  |- graph/
|  |  |- graph_algo.py
|  |  \- node.py
|  \- ui/
|     |- button.py
|     |- path_button.py
|     |- speed_button.py
|     \- viewport.py
|- test/
|  |- test_agent_play.py
|  |- test_coverage_utils.py
|  |- test_env.py
|  |- test_gameplay.py
|  |- test_gnn_lite_policy.py
|  |- test_geometry.py
|  |- test_graph.py
|  |- test_main.py
|  |- test_mediator.py
|  |- test_path.py
|  |- test_station.py
|  |- test_jongno_policy.py
|  |- test_task_runner.py
|  \- test_viewport.py
|- .gitignore
|- .pre-commit-config.yaml
|- ARCHITECTURE.md
|- environment.yml
|- GAME_RULES.md
|- INSTRUCTION_TASK_RUNNER.md
|- PROJECT_GUIDE_KR.md
|- PROGRESS.md
|- jongno_config.json
|- preprocess_jongno.py
|- pyproject.toml
|- README.md
|- visualize_smoke_log.py
\- requirements.txt
