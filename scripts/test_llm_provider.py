from careerpilot.llm_providers.factory import create_llm_client


def main() -> None:
    client = create_llm_client()

    result = client.generate(
        system_prompt="你是一个面向秋招的大模型求职助手。",
        user_prompt=(
            "请用三句话说明你如何帮助一名人工智能硕士优化大模型/Agent实习简历。"
        ),
    )

    print("\n=== LLM Response ===\n")
    print(result)


if __name__ == "__main__":
    main()
