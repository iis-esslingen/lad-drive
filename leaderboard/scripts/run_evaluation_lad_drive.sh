#!/bin/bash

BENCHMARK="$1"
export BENCHMARK

export SDL_AUDIODRIVER=dummy
export ALSA_CARD=none

export PT=$(($RANDOM % 1000 + 16000))
bash carla/CarlaUE4.sh --world-port=$PT -RenderOffScreen &

# Wait for Carla to be available
echo "Waiting for Carla to be ready on port $PT..."
RETRIES=100
until nc -z localhost $PT || [ $RETRIES -eq 0 ]; do
  echo "Still waiting for Carla... ($RETRIES retries left)"
  sleep 10
  ((RETRIES--))
done

if [ "$RETRIES" -eq 0 ]; then
    echo "Error: Carla simulator failed to start within expected time."
    exit 1
fi

echo "Carla ready, launching leaderboard_evaluator.py"

export CARLA_ROOT=carla
export CARLA_SERVER=${CARLA_ROOT}/CarlaUE4.sh
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg
export PYTHONPATH=$PYTHONPATH:leaderboard
export PYTHONPATH=$PYTHONPATH:leaderboard/team_code
export PYTHONPATH=$PYTHONPATH:scenario_runner
export PYTHONPATH=$PYTHONPATH:/beegfs/scratch/workspace/es_kafeit00-lang-traj/research-project/LADDrive/vision_encoder # TODO adjust

export LEADERBOARD_ROOT=leaderboard
export CHALLENGE_TRACK_CODENAME=SENSORS
export PORT=$PT # same as the carla server port
export TM_PORT=$(($PT+500)) # port for traffic manager, required when spawning multiple servers/clients
export DEBUG_CHALLENGE=0
export REPETITIONS=4 # multiple evaluation runs
export ROUTES=langauto/${BENCHMARK}.xml
export TEAM_AGENT=leaderboard/team_code/lad_drive_agent.py # agent
export TEAM_CONFIG=leaderboard/team_code/lad_drive_config.py # model checkpoint, not required for expert
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RESULTS_DIR="results/${TIMESTAMP}"
mkdir -p ${RESULTS_DIR}
export CHECKPOINT_ENDPOINT=${RESULTS_DIR}/${BENCHMARK}_results_lad_drive.json # results file
export SCENARIOS=leaderboard/data/official/all_towns_traffic_scenarios_public.json
export SAVE_PATH=./IMAGES/ # path for saving episodes while evaluating
export RESUME=True
# export HAS_DISPLAY=1

echo ${LEADERBOARD_ROOT}/leaderboard/leaderboard_evaluator.py
python3 -u  ${LEADERBOARD_ROOT}/leaderboard/leaderboard_evaluator.py \
  --scenarios=${SCENARIOS}  \
  --routes=${ROUTES} \
  --repetitions=${REPETITIONS} \
  --track=${CHALLENGE_TRACK_CODENAME} \
  --checkpoint=${CHECKPOINT_ENDPOINT} \
  --agent=${TEAM_AGENT} \
  --agent-config=${TEAM_CONFIG} \
  --debug=${DEBUG_CHALLENGE} \
  --record=${RECORD_PATH} \
  --resume=${RESUME} \
  --port=${PORT} \
  --trafficManagerPort=${TM_PORT}

MAX_RETRIES=10
RETRY_DELAY=30

attempt=1
while [ $attempt -le $MAX_RETRIES ]; do
    echo ">>> Attempt $attempt of $MAX_RETRIES..."

    if [ -f "$CHECKPOINT_ENDPOINT" ]; then
        total_routes=$(jq '._checkpoint.progress[1]' "$CHECKPOINT_ENDPOINT")
        completed_routes=$(jq '._checkpoint.progress[0]' "$CHECKPOINT_ENDPOINT")
    else
        total_routes=0
        completed_routes=0
    fi

    echo "Progress: $completed_routes / $total_routes routes completed"

    if [ "$completed_routes" -lt "$total_routes" ]; then
        echo "Resuming evaluation..."

        python3 -u ${LEADERBOARD_ROOT}/leaderboard/leaderboard_evaluator.py \
            --scenarios=${SCENARIOS}  \
            --routes=${ROUTES} \
            --repetitions=${REPETITIONS} \
            --track=${CHALLENGE_TRACK_CODENAME} \
            --checkpoint=${CHECKPOINT_ENDPOINT} \
            --agent=${TEAM_AGENT} \
            --agent-config=${TEAM_CONFIG} \
            --debug=${DEBUG_CHALLENGE} \
            --record=${RECORD_PATH} \
            --resume=${RESUME} \
            --port=${PORT} \
            --trafficManagerPort=${TM_PORT}
        
        echo "Sleeping $RETRY_DELAY seconds before next check..."
        sleep $RETRY_DELAY
    else
        echo "✅ All routes completed!"
        break
    fi

    ((attempt++))
done

if [ $attempt -gt $MAX_RETRIES ]; then
    echo "❌ Max retries reached. Evaluation did not complete."
    exit 1
fi
