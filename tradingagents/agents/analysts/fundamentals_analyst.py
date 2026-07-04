from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from tradingagents.agents.utils.agent_utils import (
    get_balance_sheet,
    get_cashflow,
    get_fundamentals,
    get_income_statement,
    get_instrument_context_from_state,
    get_language_instruction,
)


def create_fundamentals_analyst(llm):
    def fundamentals_analyst_node(state):
        current_date = state["trade_date"]
        instrument_context = get_instrument_context_from_state(state)

        tools = [
            get_fundamentals,
            get_balance_sheet,
            get_cashflow,
            get_income_statement,
        ]

        system_message = (
            "You are a researcher tasked with analyzing fundamental information over the past week about a company."
            + " CRITICAL — how to treat tool output:"
            + " - The financial statement data returned by tool calls is complete and correct. Do not flag missing sections, do not comment on data quality, and do not suggest data-validation steps. Do not ask for clarification."
            + " Regardless of what tool calls return — whether a single snapshot, a cash flow statement, an income statement, or all three — your output must always follow this exact pipeline and nothing else:"
            + " 1. Valuation: PE (TTM), Forward PE, PEG, Price/Book, EV/EBITDA (use the provided EV/EBITDA value directly. Do not recompute it.)"
            + " 2. Profitability: Revenue (TTM), Gross Margin, Operating Margin, Net Margin, ROE, ROA"
            + " 3. Leverage: Debt/Equity (NOTE: yfinance reports this as a percentage — 15.63 means 15.63%, not 15.63x. Label it explicitly as % in your output and interpret accordingly. A value below 50% is low leverage.), Current Ratio, Cash Position"
            + " 4. Cash Conversion: FCF (TTM), FCF 4-quarter sum if available, Capex Intensity"
            + " 5. Summary Table: all of the above as a single table"
            + " If a metric cannot be derived from the available tool output, write 'N/A — not in tool output' for that field and move on. Do not describe why it is missing. Do not ask for clarification."
            + " Never write any of the following:"
            + " - Recommendations about how to analyze the data"
            + " - Suggestions to 'monitor' or 'review' data items"
            + " - Offers to provide further analysis"
            + " - Descriptions of the data structure or column layout"
            + " - Questions or prompts directed at the reader"
            + " If you find yourself writing any of these, stop and delete it."
            + " Please write a comprehensive report of the company's fundamental information such as financial documents, company profile, basic company financials, and company financial history to gain a full view of the company's fundamental information to inform traders. Make sure to include as much detail as possible. Provide specific, actionable insights with supporting evidence to help traders make informed decisions."
            + " Make sure to append a Markdown table at the end of the report to organize key points in the report, organized and easy to read."
            + " Use the available tools: `get_fundamentals` for comprehensive company analysis, `get_balance_sheet`, `get_cashflow`, and `get_income_statement` for specific financial statements."
            + get_language_instruction()
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful AI assistant, collaborating with other assistants."
                    " Use the provided tools to progress towards answering the question."
                    " If you are unable to fully answer, that's OK; another assistant with different tools"
                    " will help where you left off. Execute what you can to make progress."
                    " If you or any other assistant has the FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** or deliverable,"
                    " prefix your response with FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** so the team knows to stop."
                    " You have access to the following tools: {tool_names}."
                    " Today's date is {current_date}; treat it as 'now' for all analysis and tool-call date ranges. {instrument_context}\n"
                    "{system_message}",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        prompt = prompt.partial(system_message=system_message)
        prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
        prompt = prompt.partial(current_date=current_date)
        prompt = prompt.partial(instrument_context=instrument_context)

        chain = prompt | llm.bind_tools(tools)

        result = chain.invoke(state["messages"])

        report = ""

        if len(result.tool_calls) == 0:
            report = result.content

        return {
            "messages": [result],
            "fundamentals_report": report,
        }

    return fundamentals_analyst_node
