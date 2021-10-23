from plume.utils import ExtendedPath, get_mongo_conn
from plume.utils.st_rerun import rerun
from uuid import uuid4
from pathlib import Path


def setup_file_state(st):
    if not hasattr(st, "state_lock"):
        # st.task_id = str(uuid4())
        task_path = ExtendedPath("preview.lck")

        def current_cursor_fn():
            return task_path.read_json()["current_cursor"]
            # if "audio_sample_idx" not in st.session_state:
            #     st.session_state.audio_sample_idx = task_path.read_json()[
            #         "current_cursor"
            #     ]
            # return st.session_state.audio_sample_idx

        def update_cursor_fn(val=0):
            task_path.write_json({"current_cursor": val})
            # rerun()
            # st.session_state.audio_sample_idx = val
            st.experimental_rerun()

        st.get_current_cursor = current_cursor_fn
        st.update_cursor = update_cursor_fn
        st.state_lock = True
        # cursor_obj = mongo_conn.find_one({"type": "current_cursor", "task_id": st.task_id})
        # if not cursor_obj:
        update_cursor_fn(0)


def setup_mongo_asr_validation_state(st):
    if not hasattr(st, "mongo_connected"):
        st.mongoclient = get_mongo_conn(col="asr_validation")
        mongo_conn = st.mongoclient
        st.task_id = str(uuid4())

        def current_cursor_fn():
            # mongo_conn = st.mongoclient
            cursor_obj = mongo_conn.find_one(
                {"type": "current_cursor", "task_id": st.task_id}
            )
            cursor_val = cursor_obj["cursor"]
            return cursor_val

        def update_cursor_fn(val=0):
            mongo_conn.find_one_and_update(
                {"type": "current_cursor", "task_id": st.task_id},
                {
                    "$set": {
                        "type": "current_cursor",
                        "task_id": st.task_id,
                        "cursor": val,
                    }
                },
                upsert=True,
            )
            rerun()

        def get_correction_entry_fn(code):
            return mongo_conn.find_one(
                {"type": "correction", "code": code}, projection={"_id": False}
            )

        def update_entry_fn(code, value):
            mongo_conn.find_one_and_update(
                {"type": "correction", "code": code},
                {"$set": {"value": value, "task_id": st.task_id}},
                upsert=True,
            )

        def set_task_fn(data_path, task_id):
            if task_id:
                st.task_id = task_id
            task_path = data_path / Path(f"task-{st.task_id}.lck")
            if not task_path.exists():
                print(f"creating task lock at {task_path}")
                task_path.touch()

        st.get_current_cursor = current_cursor_fn
        st.update_cursor = update_cursor_fn
        st.get_correction_entry = get_correction_entry_fn
        st.update_entry = update_entry_fn
        st.set_task = set_task_fn
        st.mongo_connected = True
        cursor_obj = mongo_conn.find_one(
            {"type": "current_cursor", "task_id": st.task_id}
        )
        if not cursor_obj:
            update_cursor_fn(0)
