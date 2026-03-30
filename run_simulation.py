import subprocess
import time
import sys
import os

def main():
    print("Step 1: Generating Data...")
    subprocess.run([sys.executable, "data_generator.py"], check=True)
    
    print("Step 2: Starting FL Clients...")
    client_processes = []
    ports = [8001, 8002, 8003]
    for i, port in enumerate(ports):
        # Start client i+1 on `port`
        p = subprocess.Popen([sys.executable, "client.py", "--client_id", str(i+1), "--port", str(port)])
        client_processes.append(p)
        print(f"Started Client {i+1} on port {port}")
        
    # Wait for clients to start up
    print("Waiting for clients to initialize (5s)...")
    time.sleep(5)
    
    print("Step 3: Starting FL Server Simulation...")
    try:
        subprocess.run([sys.executable, "server.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Simulation ended with error: {e}")
    except KeyboardInterrupt:
        print("\nSimulation aborted by user.")
        
    print("Step 4: Cleaning up Clients...")
    for p in client_processes:
        p.terminate()
        p.wait()
        
    print("Simulation Complete!")

if __name__ == "__main__":
    main()
