#!/bin/bash
set -eoux pipefail

exp_dir=./exp/figure4
mkdir -p $exp_dir

limit=400 # limit number of questions to speed up if needed
dataset_args="++max_num_from_same_story=5 ++split=train ++limit=$limit" # T_l dataset (400 questions)
pm_args="++correct_preference.language_model.model=preference \
            ++incorrect_preference.language_model.model=preference \
            ++correct_critique_pm.language_model.model=critique_judge \
            ++incorrect_critique_pm.language_model.model=critique_judge"

model_judge=$(python3 -m core.llm_api.registry --role judge)
judge_args="++judge.language_model.model=$model_judge ++anthropic_num_threads=$anthropic_num_threads ++organization=$organization ++limit=$limit ++judge_name=$model_judge"
score_args="++judge_name=$model_judge"

for model in $(python3 -m core.llm_api.registry --group debaters); do
    cross_examiner=$(python3 -m core.llm_api.registry --role cross_examiner)

    # 1) Just best-of-N
    for BoN in 2 4 8 16; do

        if [ $BoN -eq 1 ]; then
            temperature=0.4
        else
            temperature=0.8
        fi

        specific_args="++correct_debater.language_model.model=$model ++incorrect_debater.language_model.model=$model ++cross_examiner.language_model.model=$cross_examiner ++correct_debater.BoN=$BoN ++incorrect_debater.BoN=$BoN ++correct_debater.language_model.temperature=$temperature ++incorrect_debater.language_model.temperature=$temperature"

        # correct consultant
        python3 -m core.debate exp_dir=$exp_dir/bo${BoN}_model_${model} +experiment='consultancy' method_type='correct' $pm_args $specific_args
        python3 -m core.judge exp_dir=$exp_dir/bo${BoN}_model_${model} +experiment='consultancy' method_type='correct' $judge_args
        python3 -m core.scoring.accuracy exp_dir=$exp_dir/bo${BoN}_model_${model} +experiment='consultancy' method_type='correct' $score_args

        # incorrect consultant
        python3 -m core.debate exp_dir=$exp_dir/bo${BoN}_model_${model} +experiment='consultancy' method_type='incorrect' $pm_args $specific_args
        python3 -m core.judge exp_dir=$exp_dir/bo${BoN}_model_${model} +experiment='consultancy' method_type='incorrect' $judge_args
        python3 -m core.scoring.accuracy exp_dir=$exp_dir/bo${BoN}_model_${model} +experiment='consultancy' method_type='incorrect' $score_args
    done

    # 2) Just critique-and-refinement
    for CoN in 2 16; do

        if [ $CoN -eq 2 ]; then
            temperature=0.6
        else
            temperature=0.8
        fi

        specific_args="++correct_debater.language_model.model=$model ++incorrect_debater.language_model.model=$model ++cross_examiner.language_model.model=$cross_examiner ++correct_critic.language_model.model=$model ++incorrect_critic.language_model.model=$model ++correct_debater.BoN=1 ++incorrect_debater.BoN=1 ++correct_debater.cBoN=$CoN ++incorrect_debater.cBoN=$CoN ++correct_debater.language_model.temperature=0.4 ++incorrect_debater.language_model.temperature=0.4 ++correct_critic.language_model.temperature=$temperature ++incorrect_critic.language_model.temperature=$temperature"

        # correct consultant
        python3 -m core.debate exp_dir=$exp_dir/co${CoN}_model_${model} +experiment='consultancy_critique_story' method_type='correct' $pm_args $specific_args
        python3 -m core.judge exp_dir=$exp_dir/co${CoN}_model_${model} +experiment='consultancy_critique_story' method_type='correct' $judge_args
        python3 -m core.scoring.accuracy exp_dir=$exp_dir/co${CoN}_model_${model} +experiment='consultancy_critique_story' method_type='correct' $score_args

        # incorrect consultant
        python3 -m core.debate exp_dir=$exp_dir/co${CoN}_model_${model} +experiment='consultancy_critique_story' method_type='incorrect' $pm_args $specific_args
        python3 -m core.judge exp_dir=$exp_dir/co${CoN}_model_${model} +experiment='consultancy_critique_story' method_type='incorrect' $judge_args
        python3 -m core.scoring.accuracy exp_dir=$exp_dir/co${CoN}_model_${model} +experiment='consultancy_critique_story' method_type='incorrect' $score_args
    done

    # 3) Best-of-N and critique-and-refinement
    specific_args="++correct_debater.language_model.model=$model ++incorrect_debater.language_model.model=$model ++cross_examiner.language_model.model=$cross_examiner ++correct_critic.language_model.model=$model ++incorrect_critic.language_model.model=$model ++correct_debater.BoN=4 ++incorrect_debater.BoN=4 ++correct_debater.cBoN=8 ++incorrect_debater.cBoN=8 ++correct_debater.language_model.temperature=0.8 ++incorrect_debater.language_model.temperature=0.8 ++correct_critic.language_model.temperature=0.8 ++incorrect_critic.language_model.temperature=0.8"

    # correct consultant
    python3 -m core.debate exp_dir=$exp_dir/bo4_co8_model_${model} +experiment='consultancy_critique_story' method_type='correct' $pm_args $specific_args
    python3 -m core.judge exp_dir=$exp_dir/bo4_co8_model_${model} +experiment='consultancy_critique_story' method_type='correct' $judge_args
    python3 -m core.scoring.accuracy exp_dir=$exp_dir/bo4_co8_model_${model} +experiment='consultancy_critique_story' method_type='correct' $score_args

    # incorrect consultant
    python3 -m core.debate exp_dir=$exp_dir/bo4_co8_model_${model} +experiment='consultancy_critique_story' method_type='incorrect' $pm_args $specific_args
    python3 -m core.judge exp_dir=$exp_dir/bo4_co8_model_${model} +experiment='consultancy_critique_story' method_type='incorrect' $judge_args
    python3 -m core.scoring.accuracy exp_dir=$exp_dir/bo4_co8_model_${model} +experiment='consultancy_critique_story' method_type='incorrect' $score_args

done
