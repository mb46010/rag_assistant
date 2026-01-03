from langchain.chat_models import init_chat_model


def get_default_model():
    model = init_chat_model(
        model="gpt-4o",
        temperature=0,
        max_tokens=1000,
        # rate_limiter = ...
    )
    return model
