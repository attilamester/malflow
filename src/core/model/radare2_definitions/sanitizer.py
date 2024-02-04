import re

RE_DOT_LINE = re.compile(f'(.*)\[label="(.*)" URL=".*"];')


def sanitize_r2_bugs(ag: str):
    while ag and not ag.startswith("digraph"):
        nl = ag.find("\n")
        if nl == -1:
            break
        ag = ag[nl + 1:]

    ag_sanitized = ""
    for i, line in enumerate(ag.splitlines(keepends=True)):
        ag_sanitized += sanitize_r2_dot_line(line)
    return ag_sanitized


def sanitize_r2_dot_line(line: str):
    index_label = line.find("[label=\"")
    if index_label != -1:
        match = RE_DOT_LINE.search(line)
        label = match.group(2)
        escaped_label = label.replace("\"", "'")
        line = f"{match.group(1)}[label=\"{escaped_label}\"];"

    return line
