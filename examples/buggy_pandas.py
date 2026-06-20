# Task: Calculate average salary by department, then find the highest-paid department
import pandas as pd

df = pd.read_csv("employees.csv")
result = df.groupby("employee_id")["salary"].mean()  # BUG: should group by "department"
top = result.idxmax()
print(f"Highest paid department: {top}")
