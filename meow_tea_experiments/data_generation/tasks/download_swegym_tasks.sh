SWE_GYM_REPO=SWE-Gym/SWE-Gym
SWE_GYM_LITE_REPO=SWE-Gym/SWE-Gym-Lite
LOCAL_DATA_DIR=/root/data

hf download $SWE_GYM_REPO --include="data/*" --local-dir $LOCAL_DATA_DIR/swegym/ --repo-type dataset
hf download $SWE_GYM_LITE_REPO --include="data/*" --local-dir $LOCAL_DATA_DIR/swegym-lite/ --repo-type dataset