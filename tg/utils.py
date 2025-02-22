import re

def escape_inside_code_and_pre_tags(text: str) -> str:
    pattern = re.compile(r"<(code|pre)>(.*?)</\1>", flags=re.DOTALL)

    def replacer(match):
        tag = match.group(1)
        inner_content = match.group(2)
        escaped_content = inner_content.replace("<", "&lt;").replace(">", "&gt;")
        return f"<{tag}>{escaped_content}</{tag}>"

    return pattern.sub(replacer, text)
