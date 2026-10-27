# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/epistemic-ai/diffstory/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                              |    Stmts |     Miss |   Cover |   Missing |
|---------------------------------- | -------: | -------: | ------: | --------: |
| diffstory/\_\_init\_\_.py         |        1 |        0 |    100% |           |
| diffstory/\_\_main\_\_.py         |        2 |        2 |      0% |       3-5 |
| diffstory/analysis.py             |      938 |       65 |     93% |179-181, 264, 363, 484-485, 894, 963-970, 976-979, 1026, 1037, 1261-1262, 2203, 2253, 2307, 2318, 2321, 2326, 2691-2692, 2716-2717, 2732-2733, 2744-2745, 2874-2875, 2886-2887, 2930-2931, 2953-2954, 2988-2989, 2995-3002, 3025-3026, 3029-3030, 3039-3040, 3108-3109, 3111-3112, 3114-3115, 3133-3134 |
| diffstory/bounded\_provider.py    |       69 |       16 |     77% |55, 72-76, 84, 86-87, 94, 96, 115, 124-132 |
| diffstory/cli.py                  |      352 |       75 |     79% |48-49, 54-57, 75-77, 79-80, 123-124, 132-133, 159-160, 399, 428, 453-463, 465, 467, 474, 476, 506-520, 553-554, 583-605, 723, 733, 743-744, 785-789, 890-891, 897-898, 909, 911, 913, 984-985, 987-989, 1047-1048, 1056 |
| diffstory/compact\_packet.py      |      265 |        5 |     98% |142-154, 322, 871 |
| diffstory/evidence.py             |      150 |        3 |     98% |87, 115-116 |
| diffstory/ingest.py               |      345 |       31 |     91% |173-174, 229-230, 265-266, 296-297, 311-312, 412-413, 605-606, 610, 615-616, 623, 626-632, 755, 758, 782-783, 892-893, 896-900 |
| diffstory/models.py               |      165 |        8 |     95% |76-77, 79-80, 323-324, 365-366 |
| diffstory/packet\_schema.py       |      144 |        0 |    100% |           |
| diffstory/patches.py              |       66 |        0 |    100% |           |
| diffstory/providers.py            |      564 |      113 |     80% |147, 176, 179, 181, 278-279, 315-316, 379, 471-486, 520-521, 589-590, 613-614, 623-624, 626, 629-630, 633, 635-636, 643-644, 673-674, 692-694, 704, 714-715, 745, 842-846, 913-914, 928-930, 934-935, 938-939, 1017, 1049-1051, 1077-1078, 1119-1120, 1124-1125, 1131-1132, 1163, 1212, 1247-1250, 1329, 1339-1341, 1362-1363, 1400-1402, 1415, 1418-1419, 1427, 1434, 1438-1439, 1468-1474, 1500-1503, 1540-1544, 1547-1548, 1576-1577, 1628, 1641-1651 |
| diffstory/render.py               |      104 |       22 |     79% |55-56, 73-74, 87-88, 94-95, 111-112, 116-117, 119-120, 123-124, 151-152, 169-170, 180-181 |
| diffstory/source\_lines.py        |       11 |        0 |    100% |           |
| diffstory/usage.py                |       67 |       25 |     63% |28-39, 50, 70-84, 93, 118-119, 121-122, 168-169 |
| diffstory/walkthrough.py          |      316 |       38 |     88% |320-321, 353-354, 418-419, 422-423, 427-428, 509-510, 514-515, 523-524, 574-575, 593-594, 596-597, 606-607, 668-669, 769-770, 773-774, 816-817, 825-826, 882-883, 892-893 |
| diffstory/walkthrough\_plan.py    |      222 |       22 |     90% |384-385, 444-451, 506-517, 526, 578-579, 621-622, 770-772 |
| diffstory/walkthrough\_runtime.py |      311 |       21 |     93% |73-74, 80-81, 127, 209-214, 263-264, 324-330, 356-359, 459-460 |
| diffstory/writing.py              |        3 |        0 |    100% |           |
| **TOTAL**                         | **4095** |  **446** | **89%** |           |


## Setup coverage badge

Below are examples of the badges you can use in your main branch `README` file.

### Direct image

[![Coverage badge](https://raw.githubusercontent.com/epistemic-ai/diffstory/python-coverage-comment-action-data/badge.svg)](https://htmlpreview.github.io/?https://github.com/epistemic-ai/diffstory/blob/python-coverage-comment-action-data/htmlcov/index.html)

This is the one to use if your repository is private or if you don't want to customize anything.

### [Shields.io](https://shields.io) Json Endpoint

[![Coverage badge](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/epistemic-ai/diffstory/python-coverage-comment-action-data/endpoint.json)](https://htmlpreview.github.io/?https://github.com/epistemic-ai/diffstory/blob/python-coverage-comment-action-data/htmlcov/index.html)

Using this one will allow you to [customize](https://shields.io/endpoint) the look of your badge.
It won't work with private repositories. It won't be refreshed more than once per five minutes.

### [Shields.io](https://shields.io) Dynamic Badge

[![Coverage badge](https://img.shields.io/badge/dynamic/json?color=brightgreen&label=coverage&query=%24.message&url=https%3A%2F%2Fraw.githubusercontent.com%2Fepistemic-ai%2Fdiffstory%2Fpython-coverage-comment-action-data%2Fendpoint.json)](https://htmlpreview.github.io/?https://github.com/epistemic-ai/diffstory/blob/python-coverage-comment-action-data/htmlcov/index.html)

This one will always be the same color. It won't work for private repos. I'm not even sure why we included it.

## What is that?

This branch is part of the
[python-coverage-comment-action](https://github.com/marketplace/actions/python-coverage-comment)
GitHub Action. All the files in this branch are automatically generated and may be
overwritten at any moment.