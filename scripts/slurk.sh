#!/usr/bin/env bash
set -eu

# Call this script from the slurk-bots directory
# then build and run the chatbot individually

function errcho {
    echo "$@" 1>&2
}

function check_response {
    response=$("$@")
    if [ -z "$response" ]; then
        errcho "Unexpected error for call to: $1"
        exit 1
    fi
    echo "$response"
}

docker build --tag "slurk/concierge-bot" -f concierge/Dockerfile .

# run slurk
cd ../slurk
docker build --tag="slurk/server" -f Dockerfile .
export SLURK_DOCKER=slurk
scripts/start_server.sh
sleep 5

# create admin token
SLURK_TOKEN=$(check_response scripts/read_admin_token.sh)
echo "Admin Token:"
echo $SLURK_TOKEN

# create waiting room + layout
WAITING_ROOM=$(check_response scripts/create_default_waiting_room.sh ../slurk-bots/concierge/waiting_room_layout.json | jq .id)
echo "Waiting Room Id:"
echo $WAITING_ROOM

# create task room layout
TASK_ROOM_LAYOUT=$(check_response scripts/create_layout.sh ../101_clembench/games/chatgame/resources/task_room_layout.json | jq .id)
echo "Task Room Layout Id:"
echo $TASK_ROOM_LAYOUT

# create task
TASK_ID=$(check_response scripts/create_task.sh  "Chatbot Task" 1 "$TASK_ROOM_LAYOUT" | jq .id)
echo "Task Id:"
echo $TASK_ID

# create concierge bot
CONCIERGE_BOT_TOKEN=$(check_response scripts/create_room_token.sh $WAITING_ROOM ../slurk-bots/concierge/concierge_bot_permissions.json | jq .id | sed 's/^"\(.*\)"$/\1/')
echo "Concierge Bot Token:"
echo $CONCIERGE_BOT_TOKEN
CONCIERGE_BOT=$(check_response scripts/create_user.sh "ConciergeBot" $CONCIERGE_BOT_TOKEN | jq .id)
echo "Concierge Bot Id:"
echo $CONCIERGE_BOT
docker run -e SLURK_TOKEN="$CONCIERGE_BOT_TOKEN" -e SLURK_USER=$CONCIERGE_BOT -e SLURK_PORT=5000 --net="host" slurk/concierge-bot &
sleep 5

# create bot
BOT_TOKEN=$(check_response scripts/create_room_token.sh $WAITING_ROOM ../101_clembench/games/chatgame/resources/bot_permissions.json 10 | jq .id | sed 's/^"\(.*\)"$/\1/')
echo "Task Bot Token: "
echo $BOT_TOKEN
BOT=$(check_response scripts/create_user.sh "Chatbot" "$BOT_TOKEN" | jq .id)
echo "Bot Id:"
echo $BOT

cd ../101_clembench/

docker build --tag "slurk/chatbot" -f games/chatgame/Dockerfile .
docker run -e SLURK_TOKEN=$BOT_TOKEN -e SLURK_USER=$BOT -e SLURK_WAITING_ROOM=$WAITING_ROOM -e TASK_ID=$TASK_ID -e SLURK_PORT=5000 --net="host" slurk/chatbot &
#sleep 5

cd ../slurk/

# create user
USER=$(check_response scripts/create_room_token.sh $WAITING_ROOM ../101_clembench/games/chatgame/resources/user_permissions.json 20 $TASK_ID | jq .id | sed 's/^"\(.*\)"$/\1/')
echo "TOKEN for logging into slurk interface at localhost:5000"
echo $USER

cd ../slurk-bots/
