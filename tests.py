import requests
import time
import csv
import pandas as pd
import matplotlib.pyplot as plt

API_URL = "http://serve-sentiment-env.eba-mchs4vd2.us-east-2.elasticbeanstalk.com/predict"

# data for functional tests
test_cases = [
    {"text": "beepboop.", "expected": "FAKE"},
    {"text": "weewooweewoo", "expected": "FAKE"},
    {"text": "Today is Tuesday.", "expected": "REAL"},
    {"text": "I'm a goofy goober.", "expected": "REAL"}
]

# csv file to record latency
CSV_FILE = "latency_results.csv"

# number of repetitions per test case
REPEAT = 100

# functional tests
print("Functional Tests\n")
for case in test_cases:
    try:
        response = requests.post(API_URL, json={"message": case["text"]})
        if response.status_code == 200:
            label = response.json().get("label")
            result = "PASS" if label == case["expected"] else "FAIL"
            print(f"Input: {case['text']}\nPredicted: {label} | Expected: {case['expected']} -> {result}\n")
        else:
            print(f"Error {response.status_code}: {response.text}\n")
    except Exception as e:
        print(f"Exception for input '{case['text']}': {e}\n")

# latency tests
print("Latency Tests\n")
with open(CSV_FILE, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["test_case", "call_number", "latency_seconds"])
    
    for i, case in enumerate(test_cases, start=1):
        print(f"Running 100 calls for Test Case {i}")
        for j in range(1, REPEAT + 1):
            start_time = time.time()
            try:
                requests.post(API_URL, json={"message": case["text"]})
            except Exception as e:
                print(f"Call {j} failed: {e}")
            end_time = time.time()
            latency = end_time - start_time
            writer.writerow([i, j, latency])
            if j % 10 == 0:
                print(f"  Completed {j} calls")

# boxplot
print("\nGenerating boxplot and calculating average latency...")
df = pd.read_csv(CSV_FILE)
plt.figure(figsize=(8,6))
df.boxplot(column="latency_seconds", by="test_case")
plt.title("API Latency per Test Case")
plt.suptitle("")
plt.xlabel("Test Case")
plt.ylabel("Latency (seconds)")
plt.savefig("latency_boxplot.png")
plt.show()

# average latency per test case
avg_latency = df.groupby("test_case")["latency_seconds"].mean()
print("\nAverage latency per test case (seconds):")
print(avg_latency)
