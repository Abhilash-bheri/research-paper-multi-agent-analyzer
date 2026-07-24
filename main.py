import pdfplumber
from langgraph.graph import START,END,StateGraph
from langgraph.graph.message import add_messages
from typing import Annotated,TypedDict,Any
from pydantic import Field,BaseModel
from typing import Annotated, TypedDict, Any ,Literal,List
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage,AIMessage,SystemMessage
load_dotenv()
llm = ChatGoogleGenerativeAI(model="gemini-flash-latest")
class review_type(BaseModel):
    score:int
    approved:bool
    feedback: str
class Citation(BaseModel):
    title: str
    authors: str
    year: str

class CitationAnalysis(BaseModel):
    citations: List[Citation]
    key_papers: List[str]

class PaperAnalysis(BaseModel):
    methodology: str
    hypothesis: str
    experiments: str
    key_findings: str

class AgentState(TypedDict):

    file:Any
    text:str

    analyzerRes:PaperAnalysis

    summaryresult:str
    current_agent: str
    summary_retry: int
    feedback: review_type
    analyzer_retry: int
    citation_result: CitationAnalysis
    citation_retry: int
    final_res:str

    next_agent:str

def boss_agent(state:AgentState):

    print("BOSS Agent")

    if "text" not in state:
        return {
            "next_agent":"pdf loader"
        }

    if "analyzerRes" not in state:
        return {
            "next_agent":"paper analyzer"
        }

    if "summaryresult" not in state:
        return {
            "next_agent":"summary agent"
        }
    if "citation_result" not in state:
        return {
            "next_agent":"citation extractor"
        }

    return {
        "next_agent":"final agent"
    }

def pdf_loader(state:AgentState):
    print("PDF Loader")
    text=""
    with pdfplumber.open(state["file"]) as pdf:
        for i in pdf.pages:
            text+=i.extract_text()
    print(text)
    return {"text":text}


def paper_analyzer(state:AgentState):
    print("Paper Analyzer")
    feedback = ""

    if "feedback" in state:
        feedback = state["feedback"].feedback
    template=SystemMessage(content="""You are an expert Research Paper Analyzer.Analyze the research paper and extract:
                                    - Research methodology
                                    - Hypothesis
                                    - Experiments
                                    - Key findings
                                    Extract only information explicitly present in the paper. Do not invent or assume facts. If a requested section is unavailable, return "Not Mentioned".""")
    prompt=HumanMessage(
    content=f"""
        Research Paper:
        {state["text"]}
        Previous Review Feedback:
        {feedback}
        If feedback is provided, improve only those sections.
        """
        )
    messages=[template,prompt]
    analysis_llm=llm.with_structured_output(PaperAnalysis)
    res=analysis_llm.invoke(messages)
    return {
    "analyzerRes": res,
    "analyzer_retry": state["analyzer_retry"] + 1,
    "current_agent": "analyzer",
    }

def summary_agent(state:AgentState):
    print("summary agent")
    SUMMARY_AGENT_PROMPT = f"""
            You are an expert research paper summarization agent.

            Generate a 150-200 word executive summary from the research paper analysis.

            Include:
            - Research problem
            - Proposed methodology/approach
            - Experiments or evaluation
            - Key findings/results
            - Research significance

            Rules:
            - Use only information from the provided paper.
            - Do not hallucinate or add external knowledge.
            - Keep the summary accurate, clear, and professional.
            - If information is unavailable, mention "Not Mentioned".

            Paper Analysis:
            {state["analyzerRes"]}

            Paper Context:
            {state["text"]}

            Return only the executive summary.
            """
    res = llm.invoke(SUMMARY_AGENT_PROMPT)

    return {
    "summaryresult": res.content,
    "summary_retry": state["summary_retry"] + 1,
    "current_agent": "summary",
    }

def citation_extractor(state: AgentState):
    print("Citation Extractor")

    feedback = ""
    if "feedback" in state:
        feedback = state["feedback"].feedback

    template = SystemMessage(content="""
You are an expert academic citation extraction agent.

Extract:

1. All cited papers mentioned in the research paper.
2. Authors if available.
3. Publication year if available.
4. Organize them into a clean list.

Rules:
- Extract only citations present in the paper.
- Do not hallucinate.
- If none exist return an empty list.
Improve extraction using reviewer feedback if provided.
""")

    prompt = HumanMessage(content=f"""
Paper:

{state['text']}

Reviewer Feedback:
{feedback}
""")

    llm_struct = llm.with_structured_output(CitationAnalysis)

    res = llm_struct.invoke([template, prompt])

    return {
        "citation_result": res,
        "citation_retry": state["citation_retry"] + 1,
        "current_agent": "citation"
    }

def review_agent(state:AgentState):
    print("Review Agent")
    template = SystemMessage(content="""
        You are a strict Quality Control Reviewer for a research paper analysis system.

        Review the agent output against the original paper.
        Check accuracy, completeness, clarity, and faithfulness.
        Reject hallucinated or unsupported information.

        Provide:
        - score (1-10)
        - approved (true/false)
        - concise feedback for improvement.

        Approve only if the output is accurate and sufficiently complete.(if its even slient better not worst give more than 7)
        """)    
    if state["current_agent"] == "analyzer":
        output = state["analyzerRes"]
    elif state["current_agent"] == "citation":
        output = state["citation_result"]
    else:
        output = state["summaryresult"]

    prompt = HumanMessage(
            content=f"""
        Original Paper:
        {state['text']}

        Output to Review:
        {output}
        """
        )    
    msgs=[template,prompt]
    llm_with_struct=llm.with_structured_output(review_type)
    res=llm_with_struct.invoke(msgs)
    return {"feedback":res}

def final_agent(state:AgentState):
    print("final Boss")
    FINAL_AGENT_PROMPT = f"""
        You are the final research brief generator.
        Using the provided research paper data and agent outputs, create a complete and well-structured research brief.
        Include:
        1. Paper Overview
        - Title
        - Authors
        - Research area
        2. Research Analysis
        - Problem statement
        - Methodology
        - Hypothesis
        - Experiments
        - Key findings
        3. Executive Summary
        - Clear 150-200 word summary
        4. Key Insights
        - Main contributions
        - Practical applications
        - Limitations
        - Future scope
        Rules:
        - Use only the provided information.
        - Do not hallucinate.
        - Maintain accuracy with the original paper.
        - Present information in a professional research format.
        Original Paper:
        {state["text"]}
        Paper Analysis:
        {state["analyzerRes"]}
        Executive Summary:
        {state["summaryresult"]}
        Citations:
        {state["citation_result"]}
        Generate the final research brief.
        """
    res=llm.invoke(FINAL_AGENT_PROMPT)
    print("-----------------------------------------------------x")
    print("\n========== Final Report ==========")
    print("Research brief generated successfully.")
    print(res.content)
    return {"final_res":res.content}

def router(state: AgentState):

    feedback = state["feedback"]

    if state["current_agent"] == "analyzer":
        if feedback.score >= 7:
            return "approved"

        if state["analyzer_retry"] >= 2:
            return "approved"

        return "retry_analyzer"

    elif state["current_agent"] == "summary":
        if feedback.score >= 7:
            return "approved"

        if state["summary_retry"] >= 2:
            return "approved"

        return "retry_summary"
    elif state["current_agent"] == "citation":
        if feedback.score >= 7:
            return "approved"

        if state["citation_retry"] >= 2:
            return "approved"

        return "retry_citation"

    return "approved"
def routeBoss(state:AgentState):
    print("Boss routing")
    return state["next_agent"]

graph_builder = StateGraph(AgentState)

graph_builder.add_node("boss agent", boss_agent)
graph_builder.add_node("pdf loader", pdf_loader)
graph_builder.add_node("paper analyzer", paper_analyzer)
graph_builder.add_node("summary agent", summary_agent)
graph_builder.add_node("review agent", review_agent)
graph_builder.add_node("final agent", final_agent)
graph_builder.add_node("citation extractor",citation_extractor)

graph_builder.add_edge(START, "boss agent")

graph_builder.add_conditional_edges(
    "boss agent",
    routeBoss,
    {
        "pdf loader": "pdf loader",
        "paper analyzer": "paper analyzer",
        "summary agent": "summary agent",
        "citation extractor":"citation extractor",
        "final agent": "final agent",
    },
)

graph_builder.add_edge("pdf loader", "boss agent")
graph_builder.add_edge("paper analyzer", "review agent")
graph_builder.add_edge("summary agent", "review agent")
graph_builder.add_edge("citation extractor","review agent")

graph_builder.add_conditional_edges(
    "review agent",
    router,
    {
        "retry_analyzer": "paper analyzer",
        "retry_summary": "summary agent",
        "retry_citation":"citation extractor",
        "approved": "boss agent",
    },
)

graph_builder.add_edge("final agent", END)

graph = graph_builder.compile()

def run_workflow(file1):
    result=graph.invoke({
    "file": file1,
    "analyzer_retry": 0,
    "summary_retry": 0,
    "citation_retry": 0
    })
    return result