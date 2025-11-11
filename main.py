from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

from tools import athena_query

load_dotenv()

gemini = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash"
)


class ToolCall(BaseModel):
    name: str = Field(..., description="Name of tool call")
    input: str = Field(..., description="Input given to the tool")
    output: str = Field(..., description="Output returned by the tool")

class Response(BaseModel):
    tool_calls: list[ToolCall] = Field(..., description="List of tool calls")
    Analysis: str = Field(..., description="Analysis made from tool calls")
    Suggestions: str = Field(..., description="Suggestions for reducing emissions")

def generate_report():
    tools = [athena_query]
    agent = create_agent(
        gemini,
        tools,
        system_prompt="""
        You are a Carbon Efficiency Analyst. Your goal is to analyze our compute infrastructure's carbon footprint and provide actionable recommendations for reduction.

You have access to a tool: athena_query(query: str). This tool runs SELECT queries against an AWS Athena database.

Database Schema:

    Table: emissions

    Key Columns for Analysis:

        project_name (string): The project or model name.

        emissions (string): Total CO2eq emissions for the run (in kg). Note: This is a string, you must CAST it as double for calculations.

        timestamp (string): ISO 8601 timestamp (e.g., 2025-11-12T01:30:00Z).

        duration (double): Job run time in seconds.

        gpu_model (string): The model of the GPU used (e.g., 'NVIDIA A100', 'NVIDIA V100').

        cpu_model (string): The model of the CPU used.

        energy_consumed (string): Total energy in kWh. Note: This is a string, you must CAST it as double.

        cloud_region (string): The AWS region where the job ran (e.g., 'us-east-1').

Your Task:

Formulate and execute a series of SELECT queries using athena_query to gather insights. Your analysis must focus on two primary areas: hardware efficiency and job scheduling.

    Identify High-Impact Projects & Hardware:

        First, find the top 5 projects (project_name) contributing the most to total emissions.

        Second, for these top projects, investigate the hardware they use. Find the gpu_model or cpu_model associated with the highest average energy_consumed or emissions.

    Analyze Job Scheduling:

        Investigate when high-emission jobs are running.

        Determine if there's a concentration of jobs from top projects running during specific hours of the day (e.g., 9 AM - 5 PM). Use the timestamp column for this (you'll need to parse it, e.g., hour(from_iso8601_timestamp(timestamp))).

Final Output:

After you have gathered this data, provide a summary of your findings. Conclude with a bulleted list of specific, actionable suggestions for reducing emissions.

    Hardware Suggestion Example: "The 'legacy-model-training' project is a top 5 emitter and primarily uses 'NVIDIA V100' GPUs, which have high energy consumption. Suggestion: Prioritize migrating these workloads to 'NVIDIA A100' or 'H100' instances in the 'us-west-2' region, as they offer better performance-per-watt."

    Scheduling Suggestion Example: "The 'daily-data-processing' job accounts for 30% of emissions and runs every day at 2:00 PM. Suggestion: Schedule this batch job to run during off-peak hours (e.g., after 10:00 PM) to align with periods of higher renewable energy availability on the grid.
        """,
        response_format=ToolStrategy(Response),
        # response_format=ToolStrategy(<response_format>)
    )
    result = agent.invoke(
        {"messages": [{"role": "user","content": "Generate a report"}]},
    )

    final_response = result.get("structured_response")

    print("response:", final_response)
    return final_response





if __name__ == "__main__":
    response = generate_report()
    print(f"""
    tool_calls: {response.tool_calls}
    
    Analysis: {response.Analysis}

    Suggestions: {response.Suggestions}

    """)
