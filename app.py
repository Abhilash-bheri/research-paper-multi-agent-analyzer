import streamlit as st
from main import run_workflow

st.set_page_config(
    page_title="Research Paper Multi-Agent Analyzer",
    page_icon="📄",
    layout="wide"
)

st.markdown("""
<style>

.main{
    background:#f7f9fc;
}

h1{
    text-align:center;
    color:#1565C0;
}

.stButton>button{
    width:100%;
    height:55px;
    border-radius:10px;
    font-size:18px;
}

</style>
""", unsafe_allow_html=True)

st.title("📄 Research Paper Multi-Agent Analyzer")

st.write(
    "Upload a research paper and let multiple AI agents analyze it."
)

uploaded_file = st.file_uploader(
    "Upload PDF",
    type=["pdf"]
)

if st.button("🚀 Analyze Research Paper"):

    if uploaded_file is None:
        st.warning("Please upload a PDF.")
        st.stop()

    progress = st.progress(0)
    status = st.empty()
    workflow_box = st.empty()

    steps = [
        "📄 PDF Loader",
        "🧠 Boss Agent",
        "🔬 Paper Analyzer",
        "✅ Review Agent",
        "📝 Summary Agent",
        "📚 Citation Extractor",
        "📄 Final Report Generator"
    ]

    completed = []

    with st.spinner("Running Multi-Agent Workflow..."):

        for i, step in enumerate(steps):

            completed.append(step)

            workflow_box.markdown(
                "\n".join(
                    [
                        f"🟡 {step}" if x == step else f"✅ {x}"
                        for x in completed
                    ]
                )
            )

            status.info(step)
            progress.progress(int(((i + 1) / len(steps)) * 100))

        result = run_workflow(uploaded_file)

    progress.progress(100)
    status.success("🎉 Analysis Completed")

    workflow_box.markdown(
        "\n".join([f"✅ {x}" for x in steps])
    )

    st.divider()

    st.header("📑 Final Research Brief")

    report = result.get("final_res", "")

    # Convert different return formats to plain text
    if isinstance(report, str):
        final_report = report

    elif isinstance(report, list):
        texts = []
        for item in report:
            if isinstance(item, dict):
                texts.append(item.get("text", ""))
            else:
                texts.append(str(item))
        final_report = "\n".join(texts)

    elif hasattr(report, "content"):
        if isinstance(report.content, list):
            texts = []
            for item in report.content:
                if isinstance(item, dict):
                    texts.append(item.get("text", ""))
                else:
                    texts.append(str(item))
            final_report = "\n".join(texts)
        else:
            final_report = str(report.content)

    else:
        final_report = str(report)

    st.markdown(final_report)

    st.download_button(
        label="⬇ Download Report",
        data=final_report,
        file_name="Research_Brief.md",
        mime="text/markdown")