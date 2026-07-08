from typing import TypedDict , Annotated
from time import sleep
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph , START , END
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt , Command
from langchain_core.messages import AIMessage
from groq import RateLimitError
load_dotenv()
# Tool Create.
search_tool = TavilySearch(max_result=4)
tool = [search_tool]
tool_node = ToolNode(tool)

# Create LLM.
writer_llm = ChatGroq(model="llama-3.3-70b-versatile",temperature=0.5)
writer_llm = writer_llm.bind_tools(tool)
reviear_llm = ChatGroq(model="llama-3.3-70b-versatile",temperature=0.1)

# Create State.
class State(TypedDict):
    user_input : str
    messages : Annotated[list,add_messages]
    draft : str
    isapproved : bool
    feedback : str
    attempt : int
    
# Create Node.
def writer(state:State)->dict:
    """Writes (or rewrites) the LinkedIn post. Can call Tavily to search first."""
    writer_prompt = (
        "You are an expert LinkedIn content writer. Your job is to write "
        "engaging, professional LinkedIn posts about the given topic. "
        "If the topic requires up-to-date information, statistics, or "
        "current trends, use the web search tool to gather fresh context "
        "before writing. If you have already received feedback on a "
        "previous draft, carefully address every point in the new draft. "
        "Rules for good LinkedIn posts: strong hook in the first line, "
        "1 clear takeaway, easy to skim (short paragraphs), around "
        "150-200 words, ends with a question or call-to-action to invite "
        "engagement. Do not use hashtags."
    )
    attempt = state.get('attempt',0) + 1
    topic = state["user_input"]
    if attempt == 1:
        user_message = (
            f"Write a LinkedIn post on this topic {topic}"
            f"if you need current info search the web first "
        )
    else:
        feddback = state["feedback"]
        user_message = (
            f"your previous draft on '{topic}' was rejected"
            f"Here is the reviewer's feedback \n\n {feddback}\n\n"
            f"Write a new, improved draft that fixes every issue mentiond"
            f"do not repeat the same mistake"
        )
    message = [('system',writer_prompt),('human',user_message)]

    try:
        response = writer_llm.invoke(message)
    except RateLimitError as exc:
        print(f"\nWriter hit a Groq rate limit: {exc}")
        response = AIMessage(content=(
            "I could not generate a draft because the language model service is temporarily "
            "rate-limited. Please try again in a moment."
        ))
    except Exception as exc:
        print(f"\nWriter API call failed: {exc}")
        response = AIMessage(content=(
            "I could not generate a draft because the language model service is temporarily "
            "unavailable. Please try again in a moment."
        ))

    return {
        'attempt':attempt,
        'messages':[('human',user_message),response]
    }
    

def extract_draft(state:State)->dict:
    """After the writer finishes tool calls, pulls the final text out as the draft."""
    message = state["messages"][-1]
    draft = message.content
    print(f"\n\n generated post \n {draft} \n ")
    return {
        'draft':draft
    }
    
def human_in_the_loop(state:State)->dict:
    """Pauses the graph and waits for the human to approve or give feedback."""
    human_response = interrupt({
        'draft':state['draft'],
        'attempt':state["attempt"],
        'instruction':"Type 'approved' to accept, or type your feedback to request a rewrite."
    })
    response_human = human_response.strip()
    print("human response:", response_human)
    if response_human.lower() in ['approved','ok','yes']:
        return {
            'isapproved':True,
            'feedback':'Approved by Human.'
        }
    else:
        return {
            'isapproved':False,
            'feedback':response_human
        }
    
def stop_loop(state:State)->dict:
    if state["attempt"]>=4:
        print("reached max attempts")
        return END
    if state["isapproved"]:
        print("post haas been approved \n")
        return END
    return 'writer'

def stop_tool(state:State)->dict:
    message = state["messages"][-1]
    if getattr(message,'tool_calls',None):
        return 'tool'
    return 'extract_draft'

# Create Graph
graph = StateGraph(State)
# Add Node.
graph.add_node('writer',writer)
graph.add_node('reviwer',human_in_the_loop)
graph.add_node('extract_draft',extract_draft)
graph.add_node('tool',tool_node)
# Add Edge and Conditional Edge.
graph.add_edge(START,'writer')
graph.add_conditional_edges('writer',stop_tool,{"tool": "tool","extract_draft": "extract_draft",})
graph.add_edge('tool','writer')
graph.add_edge('extract_draft','reviwer')
graph.add_conditional_edges('reviwer',stop_loop,{'writer':'writer',END:END,})
checkpoint = MemorySaver()
# Compile
app = graph.compile(checkpointer=checkpoint)
print("=" * 55)
print("Welcome to the LinkedIn Post Generator")
print("=" * 55)
print("\nThis tool will draft a LinkedIn post for you, review it")
print("itself, and iterate until it's publish-ready.")

print("=" * 55)

topic = input("\nWhat topic do you want a LinkedIn post about?\n> ").strip()
if not topic:
    print("\nNo topic given. Exiting.")
else:
    print("\nStarting generation...\n")
    
    config = {
        "configurable":{
            "thread_id":"linkdin_session"
        }
    }

    initial_state = {
        'user_input':topic, 
        'messages' :[],
        'draft' : "",
        'isapproved' : False,
        'feedback' : "",
        'attempt' : 0
    }

    final_state = app.invoke(initial_state,config=config)

    while '__intrrupt__' in final_state:
        interrupt_data = final_state["__interrupt__"][0].value
        print("\n" + "=" * 55)
        print(f"DRAFT FOR YOUR REVIEW (Attempt {interrupt_data['attempt']})")
        print("=" * 55)
        print(interrupt_data["draft"])
        print("=" * 55)
        print(f"\n{interrupt_data['instruction']}")
        human_input = input("\nYour response: ").strip()
        final_state = app.invoke(Command(resume=human_input),config=config)

    print("\n" + "=" * 55)
    print("FINAL LINKEDIN POST")
    print("=" * 55)
    print(final_state["draft"])
    print("=" * 55)
    print(f"Total attempts: {final_state['attempt']}")
    print(f"Approved: {final_state['isapproved']}")