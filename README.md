# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/epistemic-ai/diffstory/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                              |    Stmts |     Miss |   Cover |   Missing |
|---------------------------------- | -------: | -------: | ------: | --------: |
| diffstory/\_\_init\_\_.py         |        1 |        0 |    100% |           |
| diffstory/\_\_main\_\_.py         |        2 |        2 |      0% |       3-5 |
| diffstory/analysis.py             |      938 |       65 |     93% |160-162, 245, 344, 465-466, 880, 949-956, 962-965, 1012, 1023, 1247-1248, 2189, 2239, 2293, 2304, 2307, 2312, 2677-2678, 2702-2703, 2718-2719, 2730-2731, 2860-2861, 2872-2873, 2916-2917, 2939-2940, 2974-2975, 2981-2988, 3011-3012, 3015-3016, 3025-3026, 3094-3095, 3097-3098, 3100-3101, 3119-3120 |
| diffstory/bounded\_provider.py    |       69 |       16 |     77% |55, 72-76, 84, 86-87, 94, 96, 115, 124-132 |
| diffstory/cli.py                  |      403 |       44 |     89% |57-58, 65, 132-133, 141-142, 476, 579-580, 619-633, 702-703, 705-706, 714-721, 850, 860-861, 902-906, 1021-1022, 1033, 1035, 1037, 1108-1109, 1111-1113, 1181-1182, 1190 |
| diffstory/compact\_packet.py      |      265 |        5 |     98% |142-154, 322, 871 |
| diffstory/evidence.py             |      150 |        3 |     98% |87, 115-116 |
| diffstory/ingest.py               |      357 |       29 |     92% |175-176, 231-232, 267-268, 298-299, 313-314, 612-613, 617, 622-623, 630, 633-639, 762, 765, 789-790, 899-900, 903-907 |
| diffstory/models.py               |      165 |        8 |     95% |76-77, 79-80, 323-324, 365-366 |
| diffstory/packet\_schema.py       |      144 |        0 |    100% |           |
| diffstory/patches.py              |       74 |        0 |    100% |           |
| diffstory/providers.py            |      564 |      113 |     80% |148, 177, 180, 182, 279-280, 316-317, 380, 472-487, 521-522, 590-591, 614-615, 624-625, 627, 630-631, 634, 636-637, 644-645, 674-675, 693-695, 705, 715-716, 746, 844-848, 915-916, 930-932, 936-937, 940-941, 1019, 1051-1053, 1079-1080, 1121-1122, 1126-1127, 1133-1134, 1165, 1214, 1249-1252, 1331, 1341-1343, 1364-1365, 1402-1404, 1417, 1420-1421, 1429, 1436, 1440-1441, 1470-1476, 1502-1505, 1542-1546, 1549-1550, 1578-1579, 1630, 1643-1653 |
| diffstory/render.py               |      104 |       22 |     79% |55-56, 73-74, 87-88, 94-95, 111-112, 116-117, 119-120, 123-124, 151-152, 169-170, 180-181 |
| diffstory/source\_lines.py        |       11 |        0 |    100% |           |
| diffstory/source\_selection.py    |       27 |        0 |    100% |           |
| diffstory/usage.py                |       93 |       27 |     71% |33-44, 55, 75-89, 98, 145-146, 189-190, 192-193, 243-244 |
| diffstory/walkthrough.py          |      322 |       38 |     88% |321-322, 354-355, 419-420, 423-424, 428-429, 510-511, 515-516, 524-525, 575-576, 594-595, 597-598, 607-608, 669-670, 771-772, 775-776, 803-804, 860-861, 904-905, 914-915 |
| diffstory/walkthrough\_plan.py    |      332 |       21 |     94% |496-497, 499, 560-561, 582-583, 728-729, 734, 787, 790-791, 811, 854-855, 897-898, 1046-1048 |
| diffstory/walkthrough\_runtime.py |      314 |       15 |     95% |75-76, 129, 211-216, 328-332, 358-361, 461-462 |
| diffstory/writing.py              |        3 |        0 |    100% |           |
| diffstory/writing\_evidence.py    |      123 |       10 |     92% |79-80, 179-180, 184-185, 189-190, 192-193 |
| **TOTAL**                         | **4461** |  **418** | **91%** |           |


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