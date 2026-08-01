import sys, time, importlib
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
start=time.perf_counter()
importlib.import_module("agent.agent")
print(f"import_seconds={time.perf_counter()-start:.6f}")
start=time.perf_counter()
from agent.agent import build_agent
a=build_agent(release_pointer="releases/current.json")
print(f"build_seconds={time.perf_counter()-start:.6f}")
from hospital_agent_sdk.server import create_agent_server
app=create_agent_server(test_handler=a.test)
for i in range(10):
    start=time.perf_counter(); r=app.test_client().get("/health"); print(f"health_{i}_seconds={time.perf_counter()-start:.6f} status={r.status_code}")
