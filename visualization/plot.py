import pandas as pd
from io import StringIO

csv_data = """
aclarc_label	citation_function
0	Contextualize
5	Use
1	Justify Design Choice
0	Justify Design Choice
3	Signal Gap
0	Justify Design Choice
5	Contextualize
0	Contextualize
0	Justify Design Choice
5	Use
0	Contextualize
0	Contextualize
0	Highlight Limitation
3	Signal Gap
0	Justify Design Choice
5	Use
3	Contextualize
1	Contextualize
5	Use
0	Contextualize
0	Contextualize
0	Contextualize
5	Use
0	Contextualize
3	Signal Gap
0	Contextualize
0	Highlight Limitation
0	Highlight Limitation
2	Contextualize
0	Contextualize
0	Contextualize
0	Contextualize
0	Contextualize
0	Highlight Limitation
0	Contextualize
5	Use
5	Use
0	Contextualize
5	Use
5	Use
5	Use
5	Use
0	Contextualize
0	Contextualize
5	Use
5	Use
0	Contextualize
5	Use
5	Use
5	Use
5	Use
2	Justify Design Choice
0	Signal Gap
1	Highlight Limitation
0	Contextualize
5	Use
0	Contextualize
5	Use
5	Use
5	Use
0	Signal Gap
5	Use
5	Use
4	Justify Design Choice
1	Contextualize
1	Use
5	Use
1	Use
1	Signal Gap
5	Use
0	Contextualize
0	Highlight Limitation
5	Use
4	Justify Design Choice
0	Contextualize
0	Contextualize
4	Justify Design Choice
4	Justify Design Choice
0	Contextualize
1	Modify
5	Use
0	Contextualize
0	Contextualize
0	Contextualize
5	Use
0	Contextualize
5	Use
0	Contextualize
5	Use
0	Contextualize
0	Contextualize
0	Contextualize
0	Contextualize
0	Contextualize
0	Contextualize
0	Contextualize
1	Evaluate Against
1	Evaluate Against
5	Use
5	Use
5	Use
5	Use
2	Use
0	Contextualize
5	Use
3	Contextualize
0	Contextualize
5	Use
0	Contextualize
3	Highlight Limitation
0	Contextualize
0	Contextualize
4	Justify Design Choice
1	Contextualize
5	Use
1	Use
0	Contextualize
3	Signal Gap
0	Contextualize
0	Contextualize
5	Use
3	Signal Gap
0	Contextualize
0	Contextualize
4	Justify Design Choice
2	Use
5	Use
5	Use
0	Contextualize
0	Contextualize
0	Contextualize
0	Contextualize
5	Use
0	Contextualize
1	Evaluate Against
0	Contextualize
5	Use
4	Contextualize
0	Highlight Limitation
5	Use
4	Contextualize
0	Contextualize
0	Contextualize
4	Contextualize
0	Contextualize
0	Contextualize
5	Use
0	Contextualize
0	Contextualize
0	Contextualize
4	Contextualize
0	Highlight Limitation
1	Evaluate Against
3	Signal Gap
5	Use
1	Contextualize
5	Use
0	Contextualize
0	Contextualize
0	Contextualize
5	Use
0	Contextualize
5	Use
0	Contextualize
0	Contextualize
0	Contextualize
0	Contextualize
0	Contextualize
5	Use
5	Use
5	Use
0	Contextualize
5	Use
0	Contextualize
0	Justify Design Choice
0	Contextualize
0	Contextualize
0	Contextualize
0	Contextualize
0	Contextualize
0	Contextualize
0	Contextualize
0	Highlight Limitation
0	Contextualize
0	Highlight Limitation
0	Contextualize
0	Contextualize
1	Contextualize
0	Contextualize
0	Contextualize
1	Contextualize
5	Use
0	Contextualize
0	Evaluate Against
0	Contextualize
0	Contextualize
3	Signal Gap
5	Use
0	Contextualize
1	Contextualize
0	Use
1	Contextualize
3	Contextualize
0	Justify Design Choice
1	Contextualize
1	Contextualize
0	Contextualize
0	Highlight Limitation
1	Evaluate Against
1	Contextualize
4	Justify Design Choice
4	Signal Gap
1	Contextualize
1	Contextualize
1	Contextualize
5	Use
1	Contextualize
5	Use
0	Contextualize
0	Contextualize
1	Contextualize
5	Use
1	Contextualize
5	Use
0	Contextualize
5	Use
1	Contextualize
5	Use
5	Use
1	Contextualize
1	Contextualize
5	Use
5	Use
5	Use
0	Contextualize
0	Contextualize
0	Contextualize
5	Use
5	Use
4	Justify Design Choice
0	Contextualize
0	Highlight Limitation
1	Contextualize
1	Contextualize
4	Justify Design Choice
4	Justify Design Choice
4	Justify Design Choice
1	Highlight Limitation
1	Contextualize
0	Contextualize
0	Contextualize
3	Contextualize
1	Contextualize
5	Use
0	Contextualize
0	Contextualize
2	Contextualize
0	Highlight Limitation
0	Contextualize
3	Signal Gap
0	Justify Design Choice
2	Contextualize
5	Use
5	Use
5	Use
0	Contextualize
5	Use
1	Contextualize
0	Contextualize
1	Contextualize
5	Use
1	Evaluate Against
5	Use
0	Contextualize
0	Contextualize
1	Contextualize
1	Contextualize
0	Justify Design Choice
5	Use
0	Contextualize
1	Contextualize
1	Contextualize
0	Contextualize
5	Use
0	Contextualize
5	Contextualize
0	Contextualize
0	Contextualize
0	Contextualize
5	Use
1	Contextualize
0	Contextualize
5	Use
0	Contextualize
0	Contextualize
0	Contextualize
0	Contextualize
0	Contextualize
5	Evaluate Against
0	Contextualize
0	Contextualize
5	Use
5	Use
5	Contextualize
0	Contextualize
2	Signal Gap
0	Contextualize
5	Use
0	Contextualize
5	Use
5	Use
0	Contextualize
0	Contextualize
0	Contextualize
0	Contextualize
0	Contextualize
0	Contextualize
2	Signal Gap
5	Use
1	Contextualize
0	Contextualize
1	Evaluate Against
1	Use
5	Use
0	Contextualize
0	Contextualize
0	Signal Gap
0	Contextualize
0	Contextualize
0	Contextualize
0	Contextualize
1	Highlight Limitation
0	Contextualize
0	Contextualize
1	Highlight Limitation
0	Contextualize
5	Use
0	Contextualize
0	Contextualize
1	Contextualize
0	Contextualize
0	Contextualize
0	Signal Gap
0	Contextualize
0	Contextualize
5	Use
0	Contextualize
5	Use
0	Contextualize
0	Contextualize
0	Contextualize
0	Use
0	Contextualize
5	Use
0	Highlight Limitation
0	Contextualize
0	Contextualize
5	Use
0	Highlight Limitation
0	Contextualize
0	Signal Gap
0	Contextualize
0	Contextualize
1	Highlight Limitation
0	Contextualize
0	Contextualize
5	Use
0	Contextualize
4	Justify Design Choice
0	Contextualize
3	Signal Gap
0	Contextualize
0	Contextualize
0	Contextualize
0	Contextualize
3	Signal Gap
0	Contextualize
0	Signal Gap
0	Contextualize
4	Justify Design Choice
0	Highlight Limitation
0	Contextualize
0	Contextualize
0	Contextualize
0	Use
0	Contextualize
0	Use
0	Contextualize
0	Contextualize
0	Contextualize
0	Contextualize
0	Contextualize
0	Contextualize
0	Highlight Limitation
0	Contextualize
0	Contextualize
0	Contextualize
0	Contextualize
0	Contextualize
0	Contextualize
5	Justify Design Choice
1	Contextualize
0	Contextualize
0	Highlight Limitation
0	Contextualize
5	Use
0	Contextualize
0	Contextualize
0	Contextualize
0	Contextualize
0	Contextualize
3	Use
3	Signal Gap
4	Contextualize
4	Contextualize
0	Highlight Limitation
0	Contextualize
0	Contextualize
0	Contextualize
2	Justify Design Choice
0	Contextualize
0	Contextualize
0	Contextualize
1	Contextualize
0	Contextualize
0	Contextualize
5	Use
0	Contextualize
0	Contextualize
4	Justify Design Choice
0	Contextualize
0	Use
5	Use
0	Contextualize
0	Contextualize
0	Contextualize
4	Contextualize
0	Contextualize
0	Use
0	Contextualize
0	Contextualize
0	Contextualize
0	Contextualize
0	Contextualize
0	Contextualize
5	Contextualize
0	Contextualize
3	Justify Design Choice
0	Contextualize
5	Use
0	Contextualize
3	Signal Gap
1	Contextualize
1	Justify Design Choice
0	Highlight Limitation
1	Contextualize
3	Contextualize
5	Use
5	Use
3	Signal Gap
0	Contextualize
3	Signal Gap
0	Contextualize
0	Contextualize
0	Contextualize
1	Justify Design Choice
0	Contextualize
4	Evaluate Against
5	Use
4	Evaluate Against
0	Contextualize
2	Contextualize
1	Evaluate Against
0	Contextualize
0	Justify Design Choice
0	Contextualize
0	Contextualize
0	Contextualize
1	Justify Design Choice
0	Contextualize
4	Contextualize
0	Contextualize
0	Contextualize
0	Contextualize
0	Contextualize
0	Contextualize
1	Contextualize
5	Use
0	Highlight Limitation
0	Contextualize
0	Signal Gap
5	Use
0	Contextualize
1	Contextualize
0	Use
1	Contextualize
4	Contextualize
4	Evaluate Against
0	Use
1	Contextualize
0	Contextualize
1	Contextualize
0	Contextualize
5	Contextualize
0	Contextualize
0	Contextualize
0	Contextualize
5	Use
1	Contextualize
5	Use
5	Use
0	Signal Gap
1	Contextualize
1	Justify Design Choice
0	Contextualize
0	Contextualize
0	Justify Design Choice
5	Use
0	Contextualize
5	Use
5	Use
1	Use
0	Justify Design Choice
5	Use
5	Justify Design Choice
0	Contextualize
5	Use
0	Contextualize
5	Use
5	Use
0	Contextualize
0	Contextualize
5	Contextualize
0	Contextualize
0	Contextualize
1	Use
5	Use
5	Use
4	Justify Design Choice
0	Contextualize
0	Contextualize
0	Contextualize
1	Contextualize
0	Contextualize
0	Contextualize
1	Contextualize
4	Justify Design Choice
0	Contextualize
5	Use
0	Signal Gap
1	Evaluate Against
5	Use
1	Contextualize
0	Contextualize
0	Contextualize
0	Contextualize
2	Highlight Limitation
5	Use
0	Contextualize
0	Justify Design Choice
5	Use
1	Evaluate Against
5	Use
0	Contextualize
0	Contextualize
5	Use
0	Contextualize
0	Contextualize
0	Contextualize
0	Contextualize
0	Justify Design Choice
0	Highlight Limitation
1	Highlight Limitation
1	Evaluate Against
0	Contextualize
1	Evaluate Against
0	Contextualize
3	Signal Gap
0	Contextualize
5	Use
0	Contextualize
0	Contextualize
0	Contextualize
0	Contextualize
0	Contextualize
0	Highlight Limitation
0	Contextualize
0	Justify Design Choice
0	Contextualize
0	Contextualize
1	Contextualize
0	Contextualize
0	Contextualize
0	Contextualize
0	Contextualize
0	Contextualize
0	Use
0	Contextualize
0	Contextualize
0	Contextualize
0	Contextualize
5	Use
0	Contextualize
0	Contextualize
0	Justify Design Choice
0	Highlight Limitation
5	Justify Design Choice
0	Contextualize
5	Justify Design Choice
0	Contextualize
0	Justify Design Choice
0	Contextualize
5	Use
0	Contextualize
0	Contextualize
5	Use
0	Highlight Limitation
0	Contextualize
0	Contextualize
0	Contextualize
0	Contextualize
0	Contextualize
5	Use
1	Contextualize
0	Contextualize
0	Contextualize
0	Contextualize
0	Contextualize
1	Contextualize
5	Contextualize
1	Contextualize
5	Use
0	Contextualize
0	Contextualize
0	Contextualize
0	Contextualize
5	Contextualize
5	Use
5	Use
1	Contextualize
0	Use
0	Justify Design Choice
0	Contextualize
1	Evaluate Against
5	Contextualize
1	Contextualize
1	Evaluate Against
0	Contextualize
0	Contextualize
5	Use
3	Contextualize
0	Contextualize
5	Use
0	Contextualize
5	Use
0	Contextualize
2	Contextualize
1	Highlight Limitation
0	Contextualize
0	Contextualize
1	Contextualize
0	Contextualize
1	Contextualize
3	Contextualize
1	Evaluate Against
0	Contextualize
0	Contextualize
4	Justify Design Choice
2	Contextualize
0	Contextualize
1	Evaluate Against
1	Contextualize
0	Contextualize
0	Contextualize
0	Contextualize
0	Contextualize
1	Contextualize
0	Contextualize
5	Use
0	Contextualize
0	Contextualize
4	Justify Design Choice
0	Contextualize
4	Justify Design Choice
0	Contextualize
0	Contextualize
0	Contextualize
0	Contextualize
0	Contextualize
0	Contextualize
0	Contextualize
0	Contextualize
0	Contextualize
5	Use
0	Contextualize
0	Contextualize
0	Contextualize
5	Use
0	Contextualize
0	Contextualize
5	Contextualize
0	Contextualize
4	Justify Design Choice
3	Signal Gap
3	Signal Gap
2	Highlight Limitation
2	Contextualize
2	Use
2	Use
4	Contextualize
5	Use
4	Justify Design Choice
0	Contextualize
4	Justify Design Choice
1	Contextualize
4	Contextualize
1	Contextualize
1	Contextualize
2	Use
2	Use
2	Use
0	Contextualize
0	Contextualize
2	Modify
0	Contextualize
0	Contextualize
5	Use
5	Use
1	Contextualize
0	Contextualize
0	Contextualize
0	Contextualize
0	Contextualize
0	Contextualize
5	Use
0	Contextualize
5	Use
3	Signal Gap
1	Justify Design Choice
4	Justify Design Choice
0	Contextualize
0	Contextualize
5	Use
0	Contextualize
0	Contextualize
5	Use
5	Use
5	Use
4	Highlight Limitation
0	Contextualize
0	Contextualize
0	Contextualize
5	Contextualize
0	Contextualize
0	Justify Design Choice
5	Use
0	Contextualize
0	Contextualize
0	Justify Design Choice
2	Use
4	Justify Design Choice
0	Contextualize
0	Contextualize
5	Contextualize
0	Contextualize
0	Contextualize
0	Contextualize
0	Contextualize
0	Contextualize
0	Contextualize
0	Contextualize
5	Use
0	Contextualize
0	Contextualize
0	Contextualize
5	Use
0	Contextualize
0	Contextualize
1	Evaluate Against
0	Contextualize
0	Contextualize
0	Contextualize
2	Modify
5	Use
0	Contextualize
2	Contextualize
2	Justify Design Choice
2	Use
2	Justify Design Choice
1	Contextualize
2	Justify Design Choice
2	Use
5	Use
3	Signal Gap
1	Contextualize
5	Use
0	Contextualize
1	Contextualize
0	Contextualize
0	Justify Design Choice
0	Highlight Limitation
5	Use
1	Evaluate Against
0	Contextualize
3	Signal Gap
0	Justify Design Choice
1	Evaluate Against
1	Evaluate Against
1	Contextualize
1	Evaluate Against
0	Contextualize
0	Contextualize
1	Evaluate Against
0	Contextualize
0	Contextualize
3	Signal Gap
1	Evaluate Against
5	Use
1	Evaluate Against
0	Contextualize
0	Contextualize
0	Contextualize
0	Contextualize
0	Contextualize
0	Contextualize
4	Contextualize
3	Signal Gap
0	Contextualize
4	Contextualize
0	Contextualize
0	Contextualize
5	Use
5	Use
5	Use
0	Justify Design Choice
5	Use
1	Use
2	Evaluate Against
1	Contextualize
1	Contextualize
5	Use
5	Use
1	Contextualize
1	Contextualize
1	Contextualize
1	Contextualize
1	Contextualize
1	Contextualize
2	Use
2	Contextualize
0	Contextualize
0	Contextualize
0	Contextualize
0	Contextualize
0	Contextualize
0	Contextualize
0	Contextualize
0	Contextualize
0	Contextualize
0	Contextualize
1	Contextualize
2	Contextualize
0	Contextualize
1	Contextualize
5	Use
1	Evaluate Against
0	Contextualize
0	Contextualize
0	Highlight Limitation
0	Contextualize
0	Contextualize
1	Evaluate Against
0	Contextualize
0	Contextualize
0	Contextualize
0	Highlight Limitation
0	Highlight Limitation
0	Highlight Limitation
0	Contextualize
0	Contextualize
0	Contextualize
0	Highlight Limitation
0	Contextualize
0	Highlight Limitation
4	Justify Design Choice
1	Contextualize
1	Contextualize
0	Highlight Limitation
1	Justify Design Choice
0	Highlight Limitation
1	Contextualize
1	Highlight Limitation
1	Evaluate Against
5	Use
1	Highlight Limitation
0	Justify Design Choice
3	Signal Gap
1	Evaluate Against
1	Evaluate Against
0	Contextualize
5	Modify
0	Contextualize
0	Contextualize
0	Contextualize
0	Contextualize
0	Contextualize
0	Contextualize
0	Contextualize
0	Contextualize
0	Contextualize
0	Contextualize
0	Contextualize
0	Contextualize
0	Contextualize
3	Signal Gap
0	Contextualize
0	Contextualize
0	Contextualize
0	Contextualize
0	Contextualize
0	Contextualize
0	Contextualize
0	Contextualize
0	Contextualize
0	Contextualize
0	Contextualize
4	Justify Design Choice
1	Justify Design Choice
5	Use
5	Contextualize
0	Contextualize
0	Contextualize
0	Contextualize
1	Justify Design Choice
1	Justify Design Choice
1	Evaluate Against
0	Highlight Limitation
5	Use
1	Evaluate Against
1	Justify Design Choice
1	Contextualize
1	Highlight Limitation
1	Justify Design Choice
5	Use
1	Contextualize
0	Contextualize
1	Evaluate Against
0	Contextualize
1	Signal Gap
5	Use
1	Evaluate Against
0	Contextualize
0	Contextualize
0	Contextualize
1	Contextualize
5	Signal Gap
5	Use
1	Evaluate Against
1	Contextualize
0	Contextualize
2	Contextualize
1	Contextualize
0	Contextualize
0	Contextualize
0	Contextualize
1	Contextualize
0	Justify Design Choice
1	Contextualize
0	Contextualize
0	Contextualize
1	Evaluate Against
0	Contextualize
0	Contextualize
0	Contextualize
0	Highlight Limitation
0	Contextualize
0	Contextualize
0	Contextualize
1	Contextualize
0	Contextualize
0	Contextualize
2	Modify
1	Use
1	Highlight Limitation
3	Signal Gap
0	Contextualize
5	Use
0	Contextualize
0	Contextualize
5	Use
5	Use
1	Use
0	Contextualize
5	Use
0	Contextualize
0	Contextualize
0	Contextualize
5	Use
0	Contextualize
5	Use
1	Contextualize
0	Justify Design Choice
0	Contextualize
0	Contextualize
0	Contextualize
0	Contextualize
0	Contextualize
0	Contextualize
0	Contextualize
0	Contextualize
0	Contextualize
5	Use
0	Contextualize
0	Contextualize
5	Use
5	Use
1	Contextualize
0	Contextualize
0	Contextualize
5	Use
0	Contextualize
0	Contextualize
0	Contextualize
5	Use
0	Justify Design Choice
0	Contextualize
0	Contextualize
5	Use
0	Contextualize
5	Use
0	Contextualize
0	Contextualize
1	Evaluate Against
0	Contextualize
0	Contextualize
0	Highlight Limitation
0	Contextualize
0	Contextualize
0	Contextualize
1	Contextualize
0	Contextualize
0	Contextualize
5	Use
0	Contextualize
0	Contextualize
0	Contextualize
0	Contextualize
1	Contextualize
0	Contextualize
0	Contextualize
0	Use
0	Contextualize
5	Use
5	Use
0	Contextualize
5	Justify Design Choice
0	Contextualize
0	Contextualize
0	Contextualize
0	Contextualize
0	Contextualize
5	Use
0	Contextualize
0	Contextualize
0	Contextualize
0	Contextualize
4	Justify Design Choice
0	Contextualize
0	Contextualize
5	Modify
1	Justify Design Choice
0	Contextualize
0	Contextualize
0	Contextualize
5	Use
0	Contextualize
0	Contextualize
0	Contextualize
0	Contextualize
4	Justify Design Choice
3	Signal Gap
5	Use
4	Justify Design Choice
0	Contextualize
0	Contextualize
4	Signal Gap
0	Contextualize
0	Contextualize
0	Highlight Limitation
1	Contextualize
0	Contextualize
1	Contextualize
2	Contextualize
5	Contextualize
0	Contextualize
0	Justify Design Choice
0	Contextualize
0	Contextualize
1	Contextualize
0	Contextualize
0	Contextualize
0	Contextualize
0	Justify Design Choice
0	Contextualize
0	Contextualize
0	Contextualize
0	Contextualize
0	Contextualize
0	Contextualize
0	Signal Gap
0	Justify Design Choice
0	Justify Design Choice
5	Contextualize
1	Contextualize
0	Contextualize
0	Contextualize
0	Contextualize
0	Contextualize
0	Contextualize
0	Justify Design Choice
4	Signal Gap
0	Use
0	Contextualize
0	Contextualize
0	Contextualize
0	Contextualize
0	Contextualize
0	Contextualize
0	Contextualize
0	Use
0	Contextualize
0	Contextualize
0	Justify Design Choice
0	Contextualize
0	Contextualize
0	Contextualize
0	Contextualize
0	Contextualize
1	Contextualize
0	Contextualize
0	Highlight Limitation
0	Contextualize
1	Contextualize
0	Contextualize
0	Signal Gap
0	Justify Design Choice
5	Contextualize
0	Contextualize
0	Contextualize
5	Use
1	Contextualize
0	Contextualize
1	Contextualize
0	Contextualize
0	Highlight Limitation
0	Contextualize
1	Justify Design Choice
0	Contextualize
0	Contextualize
0	Contextualize
5	Use
1	Contextualize
0	Contextualize
0	Justify Design Choice
5	Use
0	Contextualize
0	Signal Gap
0	Highlight Limitation
5	Use
3	Signal Gap
0	Contextualize
0	Contextualize
0	Contextualize
0	Highlight Limitation
5	Justify Design Choice
0	Contextualize
0	Contextualize
0	Contextualize
1	Justify Design Choice
1	Justify Design Choice
5	Use
5	Use
0	Contextualize
5	Use
2	Contextualize
3	Signal Gap
0	Contextualize
4	Contextualize
4	Contextualize
4	Contextualize
1	Justify Design Choice
5	Justify Design Choice
0	Contextualize
2	Contextualize
4	Justify Design Choice
0	Contextualize
0	Highlight Limitation
1	Contextualize
5	Contextualize
5	Use
0	Contextualize
0	Contextualize
5	Use
0	Contextualize
0	Contextualize
1	Contextualize
5	Use
5	Use
5	Use
1	Contextualize
1	Contextualize
5	Use
0	Signal Gap
1	Contextualize
1	Highlight Limitation
0	Contextualize
5	Highlight Limitation
0	Contextualize
1	Contextualize
1	Contextualize
1	Highlight Limitation
1	Highlight Limitation
0	Contextualize
5	Use
0	Contextualize
1	Highlight Limitation
0	Contextualize
0	Signal Gap
5	Use
5	Modify
5	Use
0	Contextualize
1	Contextualize
1	Contextualize
1	Contextualize
5	Use
0	Contextualize
0	Contextualize
5	Contextualize
0	Contextualize
0	Contextualize
1	Contextualize
1	Highlight Limitation
0	Contextualize
1	Contextualize
5	Use
5	Use
5	Use
0	Contextualize
5	Contextualize
1	Contextualize
4	Contextualize
5	Contextualize
0	Contextualize
0	Contextualize
5	Use
1	Highlight Limitation
5	Use
5	Use
0	Contextualize
0	Contextualize
5	Use
5	Use
4	Contextualize
1	Contextualize
5	Use
1	Highlight Limitation
0	Contextualize
5	Use
5	Use
5	Use
4	Contextualize
0	Contextualize
1	Contextualize
0	Contextualize
4	Contextualize
0	Contextualize
0	Contextualize
5	Use
1	Contextualize
5	Justify Design Choice
0	Contextualize
0	Contextualize
5	Use
4	Justify Design Choice
5	Use
5	Use
5	Use
0	Contextualize
5	Use
5	Use
5	Use
0	Contextualize
5	Use
5	Use
5	Use
1	Contextualize
1	Contextualize
5	Use
1	Highlight Limitation
0	Contextualize
4	Highlight Limitation
5	Evaluate Against
0	Contextualize
5	Use
0	Contextualize
1	Highlight Limitation
5	Evaluate Against
0	Contextualize
0	Contextualize
1	Contextualize
1	Signal Gap
0	Contextualize
1	Contextualize
2	Modify
5	Use
5	Justify Design Choice
0	Contextualize
5	Use
0	Contextualize
0	Contextualize
0	Contextualize
5	Use
1	Contextualize
0	Contextualize
2	Contextualize
0	Contextualize
1	Contextualize
0	Contextualize
1	Contextualize
3	Contextualize
1	Contextualize
1	Contextualize
0	Contextualize
1	Contextualize
1	Contextualize
1	Contextualize
0	Contextualize
1	Contextualize
1	Contextualize
1	Contextualize
0	Contextualize
1	Contextualize
1	Contextualize
1	Contextualize
1	Highlight Limitation
5	Use
0	Contextualize
0	Contextualize
1	Contextualize
0	Contextualize
3	Contextualize
1	Contextualize
1	Contextualize
0	Use
1	Contextualize
0	Contextualize
0	Contextualize
1	Contextualize
0	Contextualize
1	Contextualize
0	Contextualize
0	Contextualize
0	Contextualize
0	Contextualize
1	Contextualize
0	Contextualize
1	Contextualize
0	Contextualize
0	Contextualize
0	Contextualize
0	Contextualize
5	Use
0	Contextualize
1	Contextualize
0	Contextualize
1	Highlight Limitation
0	Contextualize
0	Contextualize
1	Highlight Limitation
0	Contextualize
3	Signal Gap
0	Contextualize
3	Signal Gap
1	Contextualize
0	Contextualize
0	Contextualize
2	Modify
5	Use
2	Contextualize
0	Justify Design Choice
0	Contextualize
3	Signal Gap
0	Contextualize
0	Contextualize
0	Contextualize
3	Justify Design Choice
0	Contextualize
5	Use
0	Contextualize
0	Contextualize
4	Contextualize
0	Contextualize
3	Signal Gap
4	Contextualize
5	Use
0	Contextualize
0	Contextualize
2	Highlight Limitation
0	Contextualize
0	Highlight Limitation
1	Contextualize
0	Contextualize
1	Contextualize
3	Signal Gap
0	Contextualize
0	Contextualize
4	Contextualize
1	Contextualize
2	Contextualize
0	Contextualize
1	Contextualize
0	Contextualize
5	Use
4	Justify Design Choice
0	Contextualize
3	Signal Gap
1	Evaluate Against
1	Contextualize
0	Contextualize
0	Contextualize
1	Highlight Limitation
0	Contextualize
1	Contextualize
1	Contextualize
0	Contextualize
1	Contextualize
3	Signal Gap
0	Justify Design Choice
0	Contextualize
4	Justify Design Choice
1	Contextualize
1	Contextualize
0	Contextualize
0	Contextualize
1	Contextualize
1	Contextualize
1	Contextualize
3	Signal Gap
1	Contextualize
0	Highlight Limitation
0	Contextualize
0	Contextualize
1	Highlight Limitation
0	Contextualize
2	Contextualize
0	Contextualize
1	Contextualize
1	Justify Design Choice
3	Signal Gap
0	Contextualize
0	Signal Gap
1	Justify Design Choice
0	Contextualize
0	Signal Gap
0	Justify Design Choice
5	Use
1	Contextualize
1	Highlight Limitation
0	Signal Gap
3	Signal Gap
0	Contextualize
1	Contextualize
1	Contextualize
0	Signal Gap
4	Justify Design Choice
0	Contextualize
1	Contextualize
2	Contextualize
0	Contextualize
1	Highlight Limitation
3	Signal Gap
1	Contextualize
0	Contextualize
2	Contextualize
1	Signal Gap
0	Contextualize
0	Contextualize
0	Contextualize
4	Contextualize
0	Contextualize
0	Use
2	Use
5	Contextualize
5	Use
0	Contextualize
1	Contextualize
5	Modify
5	Use
0	Contextualize
5	Justify Design Choice
5	Contextualize
3	Contextualize
1	Contextualize
5	Contextualize
1	Contextualize
0	Contextualize
0	Contextualize
5	Contextualize
1	Contextualize
0	Contextualize
5	Contextualize
5	Use
5	Use
5	Use
1	Contextualize
0	Contextualize
5	Use
5	Contextualize
0	Contextualize
0	Contextualize
1	Contextualize
0	Justify Design Choice
0	Contextualize
0	Contextualize
2	Use
1	Contextualize
0	Contextualize
0	Contextualize
0	Signal Gap
0	Contextualize
0	Contextualize
4	Justify Design Choice
5	Use
1	Contextualize
1	Contextualize
0	Contextualize
5	Use
1	Contextualize
0	Contextualize
5	Contextualize
0	Contextualize
0	Contextualize
1	Contextualize
0	Contextualize
0	Contextualize
1	Contextualize
0	Contextualize
2	Contextualize
1	Contextualize
0	Contextualize
0	Contextualize
0	Contextualize
1	Contextualize
5	Use
0	Signal Gap
1	Contextualize
1	Contextualize
0	Contextualize
2	Use
0	Contextualize
0	Contextualize
2	Modify
1	Contextualize
1	Contextualize
0	Contextualize
4	Justify Design Choice
1	Contextualize
4	Justify Design Choice
0	Contextualize
1	Contextualize
1	Contextualize
1	Contextualize
0	Contextualize
0	Contextualize
5	Use
0	Contextualize
1	Evaluate Against
1	Evaluate Against
5	Use
5	Use
5	Use
0	Contextualize
4	Justify Design Choice
5	Use
5	Use
5	Use
5	Use
0	Contextualize
0	Contextualize
5	Use
5	Justify Design Choice
2	Highlight Limitation
5	Use
0	Use
0	Contextualize
5	Modify
5	Evaluate Against
1	Contextualize
5	Use
1	Modify
2	Justify Design Choice
5	Use
2	Contextualize
4	Contextualize
5	Use
5	Use
5	Use
0	Contextualize
5	Justify Design Choice
1	Use
0	Contextualize
1	Contextualize
0	Contextualize
5	Use
0	Contextualize
5	Use
1	Evaluate Against
5	Use
1	Evaluate Against
0	Contextualize
2	Contextualize
2	Contextualize
2	Use
2	Use
1	Justify Design Choice
0	Contextualize
5	Contextualize
4	Justify Design Choice
0	Contextualize
5	Contextualize
3	Signal Gap
1	Use
0	Contextualize
1	Contextualize
0	Contextualize
1	Evaluate Against
0	Contextualize
1	Evaluate Against
3	Contextualize
0	Contextualize
1	Use
1	Use
0	Contextualize
3	Contextualize
0	Contextualize
3	Signal Gap
0	Contextualize
0	Contextualize
0	Justify Design Choice
0	Contextualize
1	Contextualize
0	Contextualize
0	Highlight Limitation
0	Signal Gap
0	Contextualize
0	Contextualize
0	Contextualize
1	Highlight Limitation
2	Contextualize
5	Justify Design Choice
2	Justify Design Choice
0	Contextualize
4	Justify Design Choice
5	Justify Design Choice
1	Contextualize
0	Contextualize
0	Contextualize
0	Contextualize
0	Contextualize
3	Signal Gap
0	Contextualize
0	Contextualize
0	Contextualize
0	Contextualize
0	Signal Gap
0	Justify Design Choice
0	Contextualize
0	Contextualize
0	Evaluate Against
5	Use
5	Use
0	Contextualize
0	Contextualize
0	Contextualize
4	Justify Design Choice
4	Justify Design Choice
1	Evaluate Against
2	Contextualize
2	Evaluate Against
5	Use
1	Contextualize
1	Contextualize
5	Use
1	Contextualize
5	Use
0	Justify Design Choice
0	Use
0	Highlight Limitation
0	Justify Design Choice
0	Contextualize
0	Contextualize
5	Contextualize
0	Contextualize
0	Contextualize
5	Contextualize
0	Contextualize
0	Contextualize
0	Contextualize
0	Contextualize
1	Evaluate Against
1	Use
4	Highlight Limitation
4	Contextualize
0	Justify Design Choice
1	Use
1	Highlight Limitation
1	Use
0	Contextualize
0	Contextualize
1	Contextualize
5	Use
0	Contextualize
2	Justify Design Choice
0	Contextualize
5	Contextualize
0	Contextualize
0	Contextualize
0	Justify Design Choice
0	Contextualize
0	Justify Design Choice
1	Use
5	Use
0	Contextualize
0	Contextualize
0	Contextualize
5	Use
0	Contextualize
2	Modify
1	Justify Design Choice
0	Contextualize
0	Contextualize
0	Contextualize
0	Contextualize
0	Contextualize
0	Contextualize
0	Contextualize
0	Contextualize
0	Contextualize
0	Contextualize
0	Contextualize
0	Contextualize
0	Contextualize
0	Contextualize
0	Justify Design Choice
5	Use
5	Use
0	Contextualize
0	Contextualize
5	Use
1	Contextualize
5	Use
0	Signal Gap
5	Use
0	Contextualize
1	Justify Design Choice
5	Use
1	Justify Design Choice
5	Use
0	Contextualize
5	Use
5	Use
5	Justify Design Choice
2	Use
4	Justify Design Choice
4	Justify Design Choice
4	Justify Design Choice
1	Contextualize
4	Modify
4	Justify Design Choice
0	Contextualize
5	Use
0	Signal Gap
0	Contextualize
0	Signal Gap
0	Contextualize
0	Contextualize
0	Contextualize
0	Contextualize
0	Contextualize
0	Contextualize
0	Signal Gap
5	Use
5	Justify Design Choice
0	Contextualize
0	Contextualize
0	Contextualize
0	Contextualize
0	Contextualize
3	Signal Gap
0	Contextualize
0	Contextualize
0	Contextualize
5	Use
0	Contextualize
0	Contextualize
0	Contextualize
1	Justify Design Choice
0	Contextualize
0	Contextualize
0	Contextualize
0	Contextualize
0	Contextualize
0	Contextualize
5	Use
1	Justify Design Choice
5	Modify
0	Contextualize
0	Contextualize
0	Contextualize
0	Contextualize
0	Contextualize
1	Justify Design Choice
0	Contextualize
0	Contextualize
1	Contextualize
0	Contextualize
0	Contextualize
0	Contextualize
0	Contextualize
0	Contextualize
0	Contextualize
0	Contextualize
3	Signal Gap
0	Contextualize
0	Highlight Limitation
1	Evaluate Against
0	Contextualize
0	Contextualize
3	Signal Gap
0	Signal Gap
1	Evaluate Against
4	Justify Design Choice
1	Signal Gap
3	Signal Gap
3	Signal Gap
5	Use
1	Evaluate Against
4	Justify Design Choice
1	Contextualize
4	Justify Design Choice
1	Contextualize
4	Justify Design Choice
1	Evaluate Against
0	Contextualize
2	Use
1	Highlight Limitation
3	Signal Gap
3	Signal Gap
5	Use
4	Justify Design Choice
5	Use
2	Use
5	Use
5	Use
3	Signal Gap
5	Use
5	Use
1	Evaluate Against
0	Evaluate Against
5	Highlight Limitation
1	Contextualize
1	Justify Design Choice
0	Contextualize
1	Contextualize
0	Contextualize
0	Contextualize
1	Highlight Limitation
0	Contextualize
0	Contextualize
5	Use
1	Contextualize
5	Use
4	Justify Design Choice
1	Contextualize
1	Contextualize
1	Evaluate Against
5	Use
1	Evaluate Against
1	Contextualize
3	Highlight Limitation
3	Signal Gap
0	Contextualize
1	Justify Design Choice
0	Contextualize
3	Signal Gap
0	Contextualize
0	Contextualize
0	Contextualize
0	Contextualize
1	Evaluate Against
1	Evaluate Against
1	Evaluate Against
0	Justify Design Choice
0	Evaluate Against
0	Contextualize
5	Use
1	Highlight Limitation
2	Evaluate Against
1	Contextualize
2	Modify
2	Justify Design Choice
2	Modify
1	Justify Design Choice
1	Justify Design Choice
5	Use
5	Use
5	Use
5	Use
1	Evaluate Against
0	Contextualize
5	Use
5	Use
5	Contextualize
0	Contextualize
0	Contextualize
0	Contextualize
1	Evaluate Against
1	Evaluate Against
0	Contextualize
5	Modify
"""

data = pd.read_csv(StringIO(csv_data), sep="\t")

data['aclarc_label'] = data['aclarc_label'].map({
    0: 'BACKGROUND',
    1: 'COMPARES_CONTRASTS',
    2: 'EXTENSION',
    3: 'FUTURE',
    4: 'MOTIVATION',
    5: 'USES'
})

import plotly.graph_objects as go

def plot_sankey(df: pd.DataFrame, col1: str, col2: str, title: str = "Category Change Sankey Diagram"):
    """
    Generates and displays a Sankey diagram visualizing the flow between
    categories in two columns of a Pandas DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame.
        col1 (str): The name of the source column (first schema).
        col2 (str): The name of the target column (second schema).
        title (str): The title for the Sankey diagram.
    """
    # --- 1. Data Preparation ---
    # Ensure columns exist
    if col1 not in df.columns or col2 not in df.columns:
        raise ValueError(f"Columns '{col1}' and/or '{col2}' not found in DataFrame.")

    # Drop rows where either column has NaN values, as they cannot be plotted
    df_clean = df[[col1, col2]].dropna().copy()
    if len(df_clean) == 0:
        print("Warning: No valid data points after removing NaNs. Cannot generate Sankey.")
        return

    # Convert categories to string to ensure consistent handling
    df_clean[col1] = df_clean[col1].astype(str)
    df_clean[col2] = df_clean[col2].astype(str)

    # --- 2. Create Nodes (Labels) ---
    # Combine labels ensuring uniqueness and order. We add suffixes to distinguish
    # labels that might be identical but belong to different columns visually,
    # although Plotly handles this internally with indices.
    # More robust approach: create a single list of unique labels across both columns.
    all_labels = sorted(list(pd.concat([df_clean[col1], df_clean[col2]]).unique()))
    label_to_id = {label: i for i, label in enumerate(all_labels)}

    # --- 3. Create Links (Source, Target, Value) ---
    # Calculate the flow counts between categories
    link_data = df_clean.groupby([col1, col2]).size().reset_index(name='value')

    # Map category names to their numerical IDs for Plotly
    link_data['source'] = link_data[col1].map(label_to_id)
    link_data['target'] = link_data[col2].map(label_to_id)

    # Filter out any potential NaN links if mapping failed (shouldn't happen with current logic)
    link_data = link_data.dropna(subset=['source', 'target'])
    link_data['source'] = link_data['source'].astype(int)
    link_data['target'] = link_data['target'].astype(int)

    # --- 4. Create Sankey Diagram ---
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,               # Vertical padding between nodes
            thickness=20,         # Thickness of the nodes
            line=dict(color="black", width=0.5), # Border line
            label=all_labels,     # Node labels (unique categories)
            # Optional: Add colors for nodes if desired
            # color=["blue", "red", "green", ...] # List of colors matching all_labels
        ),
        link=dict(
            source=link_data['source'].tolist(),  # List of source node indices
            target=link_data['target'].tolist(),  # List of target node indices
            value=link_data['value'].tolist(),    # List of flow values (counts)
            # Optional: Add colors for links if desired
            # color = ["rgba(0,0,255,0.2)", ...] # List of colors matching links
        )
    )])

    # --- 5. Customize Layout and Display ---
    fig.update_layout(
        title_text=title,
        font_size=12,
        # Increase height if many categories
        # height=600 + len(all_labels) * 10
    )
    fig.show()

plot_sankey(data, 'aclarc_label', 'citation_function', title="Flow between Schema1 and Schema2 Categories")
