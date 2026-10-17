#!/bin/bash
set -eoux pipefail

exp_dir=./exp/figure1

limit=47 # limit number of questions to speed up if needed
dataset_args="++max_num_from_same_story=5 ++split=both ++human_experiments=8 ++limit=$limit" # T_h dataset (47 questions)
debate_args="++correct_debater.language_model.model=debater \
            ++incorrect_debater.language_model.model=debater \
            ++correct_preference.language_model.model=preference \
            ++incorrect_preference.language_model.model=preference \
            ++correct_debater.BoN=16 ++incorrect_debater.BoN=16 \
            ++correct_debater.language_model.temperature=0.8 \
            ++incorrect_debater.language_model.temperature=0.8"
mkdir -p $exp_dir

# Run debate doesn't need interactive judge
python3 -m core.debate exp_dir=$exp_dir +experiment='debate' $debate_args $dataset_args

# Initialise additional baselines
python3 -m core.debate exp_dir=$exp_dir +experiment='blind' $dataset_args
python3 -m core.debate exp_dir=$exp_dir +experiment='oracle' $dataset_args

for model_judge in $(python3 -m core.llm_api.registry --group judges); do

    # Run debate and consultancy with interactive judge
    interactive_args="$debate_args ++cross_examiner.language_model.model=$model_judge"
    python3 -m core.debate exp_dir=$exp_dir/interactive=$model_judge +experiment='debate_interactive' \
        $interactive_args $dataset_args
    python3 -m core.debate exp_dir=$exp_dir/interactive=$model_judge +experiment='consultancy' method_type='correct' \
        $interactive_args $dataset_args
    python3 -m core.debate exp_dir=$exp_dir/interactive=$model_judge +experiment='consultancy' method_type='incorrect' \
        $interactive_args $dataset_args

    # Judge all experiments
    judge_args="++judge.language_model.model=$model_judge ++judge_name=$model_judge"
    python3 -m core.judge exp_dir=$exp_dir +experiment='debate' \
        $judge_args
    python3 -m core.judge exp_dir=$exp_dir/interactive=$model_judge +experiment='debate_interactive' \
        $judge_args
    python3 -m core.judge exp_dir=$exp_dir/interactive=$model_judge +experiment='consultancy' method_type='correct' \
        $judge_args
    python3 -m core.judge exp_dir=$exp_dir/interactive=$model_judge +experiment='consultancy' method_type='incorrect' \
        $judge_args

    # Additional baselines
    python3 -m core.judge exp_dir=$exp_dir +experiment='blind' \
        $judge_args

    model_judge_expert=$model_judge
    judge_expert_args="++judge.language_model.model=$model_judge_expert ++judge_name=$model_judge"
    python3 -m core.judge exp_dir=$exp_dir +experiment='oracle' \
        $judge_expert_args

    # Score all experiments
    score_args="++judge_name=$model_judge"
    python3 -m core.scoring.accuracy exp_dir=$exp_dir/interactive=$model_judge +experiment='debate_interactive' $score_args
    python3 -m core.scoring.accuracy exp_dir=$exp_dir/interactive=$model_judge +experiment='consultancy' method_type='correct' $score_args
    python3 -m core.scoring.accuracy exp_dir=$exp_dir/interactive=$model_judge +experiment='consultancy' method_type='incorrect' $score_args
    python3 -m core.scoring.accuracy exp_dir=$exp_dir +experiment='blind' $score_args
    python3 -m core.scoring.accuracy exp_dir=$exp_dir +experiment='oracle' $score_args
done
