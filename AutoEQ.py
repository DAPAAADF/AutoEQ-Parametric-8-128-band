
import argparse
import sys
import os
import numpy as np
from scipy import interpolate, optimize, ndimage

try:
    from tqdm import tqdm as _tqdm
    def _progress(iterable, **kw): return _tqdm(iterable, **kw)
except ImportError:
    def _progress(iterable, **kw): return iterable  # graceful fallback


_BUILTIN_TARGET = [
    (20.0000000000,59.0000000000),
    (20.2909066988,59.0000000000),
    (20.5860447329,59.0000000000),
    (20.8854756485,59.0000000000),
    (21.1892618872,59.0000000000),
    (21.4974667984,59.0000000000),
    (21.8101546533,59.0000000000),
    (22.1273906578,59.0000000000),
    (22.4492409662,59.0000000000),
    (22.7757726951,59.0000000000),
    (23.1070539374,59.0000000000),
    (23.4431537764,59.0000000000),
    (23.7841423001,59.0000000000),
    (24.1300906160,59.0000000000),
    (24.4810708661,59.0000000000),
    (24.8371562415,59.0000000000),
    (25.1984209979,59.0000000000),
    (25.5649404712,59.0000000000),
    (25.9367910930,59.0000000000),
    (26.3140504067,59.0000000000),
    (26.6967970834,59.0000000000),
    (27.0851109387,59.0000000000),
    (27.4790729492,59.0000000000),
    (27.8787652690,59.0000000000),
    (28.2842712475,59.0000000000),
    (28.6956754462,59.0000000000),
    (29.1130636568,59.0000000000),
    (29.5365229188,59.0000000000),
    (29.9661415375,59.0000000000),
    (30.4020091030,59.0000000000),
    (30.8442165082,59.0000000000),
    (31.2928559682,59.0000000000),
    (31.7480210394,59.0000000000),
    (32.2098066390,59.0000000000),
    (32.6783090648,59.0000000000),
    (33.1536260154,59.0000000000),
    (33.6358566101,59.0000000000),
    (34.1251014105,59.0000000000),
    (34.6214624402,59.0000000000),
    (35.1250432075,59.0000000000),
    (35.6359487256,59.0000000000),
    (36.1542855356,59.0000000000),
    (36.6801617282,59.0000000000),
    (37.2136869661,59.0000000000),
    (37.7549725073,59.0000000000),
    (38.3041312279,59.0000000000),
    (38.8612776461,59.0000000000),
    (39.4265279456,59.0000000000),
    (40.0000000000,59.0000000000),
    (40.5818133975,59.0000000000),
    (41.1720894657,59.0000000000),
    (41.7709512971,59.0000000000),
    (42.3785237744,59.0000000000),
    (42.9949335968,59.0000000000),
    (43.6203093066,59.0000000000),
    (44.2547813156,59.0000000000),
    (44.8984819324,59.0000000000),
    (45.5515453903,59.0000000000),
    (46.2141078749,59.0000000000),
    (46.8863075528,59.0000000000),
    (47.5682846001,59.0000000000),
    (48.2601812320,59.0000000000),
    (48.9621417322,59.0000000000),
    (49.6743124829,59.0000000000),
    (50.3968419958,59.0000000000),
    (51.1298809424,59.0000000000),
    (51.8735821860,59.0000000000),
    (52.6281008133,59.0000000000),
    (53.3935941668,59.0000000000),
    (54.1702218775,59.0000000000),
    (54.9581458983,59.0000000000),
    (55.7575305380,59.0000000000),
    (56.5685424949,59.0000000000),
    (57.3913508924,59.0000000000),
    (58.2261273137,59.0000000000),
    (59.0730458376,59.0000000000),
    (59.9322830751,59.0000000000),
    (60.8040182060,59.0000000000),
    (61.6884330163,59.0000000000),
    (62.5857119363,59.0000000000),
    (63.4960420787,59.0000000000),
    (64.4196132780,59.0000000000),
    (65.3566181296,59.0000000000),
    (66.3072520307,59.0000000000),
    (67.2717132203,59.0000000000),
    (68.2502028209,59.0000000000),
    (69.2429248805,59.0000000000),
    (70.2500864149,58.9969152034),
    (71.2718974512,58.9212251030),
    (72.3085710713,58.7483276502),
    (73.3603234564,58.4926389137),
    (74.4273739322,58.1789270410),
    (75.5099450145,57.8404804506),
    (76.6082624559,57.5159732985),
    (77.7225552923,57.2452217758),
    (78.8530558912,57.0642170515),
    (80.0000000000,57.0000000000),
    (81.1636267950,57.0000000000),
    (82.3441789315,57.0000000000),
    (83.5419025942,57.0000000000),
    (84.7570475487,57.0000000000),
    (85.9898671937,57.0000000000),
    (87.2406186132,57.0000000000),
    (88.5095626311,57.0000000000),
    (89.7969638648,57.0000000000),
    (91.1030907805,57.0000000000),
    (92.4282157498,57.0000000000),
    (93.7726151055,57.0000000000),
    (95.1365692002,57.0000000000),
    (96.5203624640,57.0000000000),
    (97.9242834644,57.0000000000),
    (99.3486249659,57.0000000000),
    (100.7936839916,57.0000000000),
    (102.2597618848,57.0000000000),
    (103.7471643721,57.0000000000),
    (105.2562016267,57.0000000000),
    (106.7871883336,57.0000000000),
    (108.3404437550,57.0000000000),
    (109.9162917966,57.0000000000),
    (111.5150610759,57.0000000000),
    (113.1370849898,57.0000000000),
    (114.7827017849,57.0000000000),
    (116.4522546274,57.0000000000),
    (118.1460916752,57.0000000000),
    (119.8645661501,57.0000000000),
    (121.6080364119,57.0000000000),
    (123.3768660326,57.0000000000),
    (125.1714238726,57.0000000000),
    (126.9920841575,57.0000000000),
    (128.8392265559,57.0000000000),
    (130.7132362593,57.0000000000),
    (132.6145040614,57.0000000000),
    (134.5434264406,57.0000000000),
    (136.5004056418,57.0000000000),
    (138.4858497610,57.0000000000),
    (140.5001728299,57.0000000000),
    (142.5437949025,57.0000000000),
    (144.6171421426,57.0000000000),
    (146.7206469127,57.0000000000),
    (148.8547478643,57.0000000000),
    (151.0198900291,57.0568174842),
    (153.2165249118,57.5461525138),
    (155.4451105846,58.4569708374),
    (157.7061117824,59.6078863624),
    (160.0000000000,60.7500000000),
    (162.3272535900,61.6184316056),
    (164.6883578630,61.9946766453),
    (167.0838051884,62.0000000000),
    (169.5140950975,62.0000000000),
    (171.9797343873,62.0000000000),
    (174.4812372264,62.0000000000),
    (177.0191252622,62.0000000000),
    (179.5939277295,62.0000000000),
    (182.2061815611,62.0000000000),
    (184.8564314996,62.0000000000),
    (187.5452302111,62.0000000000),
    (190.2731384004,62.0000000000),
    (193.0407249281,62.0000000000),
    (195.8485669287,62.0000000000),
    (198.6972499318,62.0000000000),
    (201.5873679832,62.0000000000),
    (204.5195237697,62.0000000000),
    (207.4943287442,62.0000000000),
    (210.5124032534,62.0000000000),
    (213.5743766672,62.0000000000),
    (216.6808875099,62.0000000000),
    (219.8325835933,62.0000000000),
    (223.0301221518,62.0000000000),
    (226.2741699797,62.0000000000),
    (229.5654035698,62.0000000000),
    (232.9045092548,62.0000000000),
    (236.2921833503,62.0000000000),
    (239.7291323003,62.0000000000),
    (243.2160728239,62.0000000000),
    (246.7537320653,62.0000000000),
    (250.3428477452,62.0000000000),
    (253.9841683149,62.0000000000),
    (257.6784531119,62.0000000000),
    (261.4264725186,62.0000000000),
    (265.2290081229,62.0000000000),
    (269.0868528812,62.0000000000),
    (273.0008112836,62.0000000000),
    (276.9716995220,62.0000000000),
    (281.0003456597,62.0000000000),
    (285.0875898049,62.0000000000),
    (289.2342842852,62.0000000000),
    (293.4412938255,62.0000000000),
    (297.7094957287,62.0000000000),
    (302.0397800581,62.0000000000),
    (306.4330498235,62.0000000000),
    (310.8902211692,62.0000000000),
    (315.4122235649,62.0000000000),
    (320.0000000000,62.0000000000),
    (324.6545071800,62.0000000000),
    (329.3767157259,62.0000000000),
    (334.1676103768,62.0000000000),
    (339.0281901950,62.0000000000),
    (343.9594687746,62.0000000000),
    (348.9624744529,62.0000000000),
    (354.0382505244,61.9910637593),
    (359.1878554590,61.9538560254),
    (364.4123631221,61.8869689975),
    (369.7128629991,61.7899373738),
    (375.0904604222,61.6626913768),
    (380.5462768009,61.5055980642),
    (386.0814498562,61.3194995272),
    (391.6971338575,61.1057466583),
    (397.3944998635,60.8662270545),
    (403.1747359664,60.6033855178),
    (409.0390475393,60.3202355416),
    (414.9886574883,60.0203601269),
    (421.0248065068,59.7079002698),
    (427.1487533344,59.3875295120),
    (433.3617750198,59.0644130541),
    (439.6651671866,58.7441501110),
    (446.0602443037,58.4326984468),
    (452.5483399594,58.1362803731),
    (459.1308071395,57.8612699330),
    (465.8090185095,57.6140615367),
    (472.5843667007,57.4009209496),
    (479.4582646005,57.2278202776),
    (486.4321456477,57.1002594221),
    (493.5074641306,57.0230773886),
    (500.6856954905,57.0086924801),
    (507.9683366298,58.0293849082),
    (515.3569062238,59.6184260941),
    (522.8529450372,60.0000000000),
    (530.4580162458,60.0000000000),
    (538.1737057624,60.0000000000),
    (546.0016225673,60.0000000000),
    (553.9433990439,60.0000000000),
    (562.0006913195,60.0000000000),
    (570.1751796098,60.0000000000),
    (578.4685685703,60.0000000000),
    (586.8825876510,60.0000000000),
    (595.4189914574,60.0000000000),
    (604.0795601163,60.0000000000),
    (612.8660996471,60.0000000000),
    (621.7804423383,60.0000000000),
    (630.8244471297,60.0000000000),
    (640.0000000000,60.0000000000),
    (649.3090143600,60.0000000000),
    (658.7534314519,60.0000000000),
    (668.3352207536,60.0000000000),
    (678.0563803900,60.0000000000),
    (687.9189375493,60.0000000000),
    (697.9249489058,60.0000000000),
    (708.0765010489,60.0000000000),
    (718.3757109180,60.0000000000),
    (728.8247262443,60.0000000000),
    (739.4257259983,60.0000000000),
    (750.1809208443,60.0000000000),
    (761.0925536018,60.0000000000),
    (772.1628997124,60.0000000000),
    (783.3942677150,60.0000000000),
    (794.7889997271,60.0000000000),
    (806.3494719327,60.0000000000),
    (818.0780950787,60.0000000000),
    (829.9773149767,60.0000000000),
    (842.0496130136,60.0000000000),
    (854.2975066688,60.0000000000),
    (866.7235500396,60.0000000000),
    (879.3303343732,60.0000000000),
    (892.1204886074,60.0000000000),
    (905.0966799188,60.0000000000),
    (918.2616142791,60.0000000000),
    (931.6180370190,60.0000000000),
    (945.1687334013,60.0000000000),
    (958.9165292011,60.0000000000),
    (972.8642912955,60.0000000000),
    (987.0149282611,60.0000000000),
    (1001.3713909810,60.0000928089),
    (1015.9366732597,60.0125228212),
    (1030.7138124476,60.0464075854),
    (1045.7058900743,60.1023828684),
    (1060.9160324916,60.1808942027),
    (1076.3474115248,60.2821721516),
    (1092.0032451345,60.4062077525),
    (1107.8867980879,60.5527285956),
    (1124.0013826389,60.7211760675),
    (1140.3503592197,60.9106843489),
    (1156.9371371406,61.1200618224),
    (1173.7651753020,61.3477756044),
    (1190.8379829148,61.5919399707),
    (1208.1591202326,61.8503094875),
    (1225.7321992942,62.1202776948),
    (1243.5608846767,62.3988822043),
    (1261.6488942595,62.6828170731),
    (1280.0000000000,62.9684532865),
    (1298.6180287201,63.2518681304),
    (1317.5068629037,63.5288841469),
    (1336.6704415071,63.7951182419),
    (1356.1127607799,64.0460413522),
    (1375.8378750985,64.2770488655),
    (1395.8498978116,64.4835417362),
    (1416.1530020977,64.6610179283),
    (1436.7514218360,64.8051734637),
    (1457.6494524886,64.9120119497),
    (1478.8514519966,64.9779610077),
    (1500.3618416887,65.0000044869),
    (1522.1851072035,65.0168477506),
    (1544.3257994247,65.0670301328),
    (1566.7885354300,65.1513145368),
    (1589.5779994541,65.2699812974),
    (1612.6989438655,65.4227738956),
    (1636.1561901574,65.6088489279),
    (1659.9546299533,65.8267321832),
    (1684.0992260271,66.0742828358),
    (1708.5950133377,66.3486679016),
    (1733.4471000793,66.6463491866),
    (1758.6606687464,66.9630850013),
    (1784.2409772148,67.2939488789),
    (1810.1933598376,67.6333674287),
    (1836.5232285581,67.9751792504),
    (1863.2360740381,68.3127165209),
    (1890.3374668026,68.6389104345),
    (1917.8330584022,68.9464211113),
    (1945.7285825909,69.2277918856),
    (1974.0298565222,69.4756270437),
    (2002.7427819620,69.6827910973),
    (2031.8733465194,69.8426265644),
    (2061.4276248951,69.9491860107),
    (2091.4117801487,69.9974727955),
    (2121.8320649832,70.2729008123),
    (2152.6948230496,70.6586852881),
    (2184.0064902691,71.0500811284),
    (2215.7735961758,71.4471699522),
    (2248.0027652779,71.8500345660),
    (2280.7007184393,72.2587589805),
    (2313.8742742813,72.6734284285),
    (2347.5303506040,73.0941293825),
    (2381.6759658296,73.5209495729),
    (2416.3182404652,73.9539780058),
    (2451.4643985884,74.3933049824),
    (2487.1217693533,74.8390221169),
    (2523.2977885190,75.0000000000),
    (2560.0000000001,75.0000000000),
    (2597.2360574402,75.0000000000),
    (2635.0137258074,75.0000000000),
    (2673.3408830143,75.0000000000),
    (2712.2255215599,75.0000000000),
    (2751.6757501971,75.0000000000),
    (2791.6997956232,75.0000000000),
    (2832.3060041955,75.0000000000),
    (2873.5028436721,75.0000000000),
    (2915.2989049772,75.0000000000),
    (2957.7029039931,75.0000000000),
    (3000.7236833774,75.0000000000),
    (3044.3702144071,75.0000000000),
    (3088.6515988494,75.0000000000),
    (3133.5770708600,75.0000000000),
    (3179.1559989082,75.0000000000),
    (3225.3978877310,75.0000000000),
    (3272.3123803148,75.0000000000),
    (3319.9092599067,75.0000000000),
    (3368.1984520542,75.0000000000),
    (3417.1900266754,75.0000000000),
    (3466.8942001586,75.0000000000),
    (3517.3213374928,75.0000000000),
    (3568.4819544296,75.0000000000),
    (3620.3867196753,75.0000000000),
    (3673.0464571163,75.0000000000),
    (3726.4721480761,75.0000000000),
    (3780.6749336053,75.0000000000),
    (3835.6661168044,74.9956438447),
    (3891.4571651819,74.9714722881),
    (3948.0597130445,74.9258115063),
    (4005.4855639241,74.8587557598),
    (4063.7466930387,74.7709357849),
    (4122.8552497902,74.6635569368),
    (4182.8235602974,74.5384247051),
    (4243.6641299665,74.3979547239),
    (4305.3896460992,74.2451644613),
    (4368.0129805382,74.0836439795),
    (4431.5471923516,73.9175035053),
    (4496.0055305558,73.7512960888),
    (4561.4014368787,73.5899143465),
    (4627.7485485626,73.4384612097),
    (4695.0607012081,73.3020957176),
    (4763.3519316592,73.1858562080),
    (4832.6364809305,73.0944647266),
    (4902.9287971769,73.0321180733),
    (4974.2435387067,73.0022725586),
    (5046.5955770380,72.9689362820),
    (5120.0000000002,72.9200000000),
    (5194.4721148803,72.8703519234),
    (5270.0274516149,72.8199816989),
    (5346.6817660286,72.7688788226),
    (5424.4510431198,72.7170326379),
    (5503.3515003942,72.6644323331),
    (5583.3995912463,72.6110669392),
    (5664.6120083911,72.5569253277),
    (5747.0056873442,72.5019962084),
    (5830.5978099545,72.4462681267),
    (5915.4058079863,72.3897294613),
    (6001.4473667548,72.3323684222),
    (6088.7404288142,72.2741730475),
    (6177.3031976989,72.2151312015),
    (6267.1541417201,72.1552305722),
    (6358.3119978165,72.0944586681),
    (6450.7957754620,72.0328028164),
    (6544.6247606297,71.9702501596),
    (6639.8185198134,71.9067876535),
    (6736.3969041085,71.8424020639),
    (6834.3800533509,71.7770799644),
    (6933.7884003172,71.7108077331),
    (7034.6426749857,71.6435715500),
    (7136.9639088592,71.5753573941),
    (7240.7734393506,71.5061510404),
    (7346.0929142326,71.4359380572),
    (7452.9442961523,71.3647038026),
    (7561.3498672106,71.2924334219),
    (7671.3322336089,71.2191118443),
    (7782.9143303638,71.1447237798),
    (7896.1194260890,71.0692537159),
    (8010.9711278481,70.9926859148),
    (8127.4933860775,70.9150044093),
    (8245.7104995805,70.8361930003),
    (8365.6471205948,70.7562352529),
    (8487.3282599330,70.6751144934),
    (8610.7792921984,70.5928138052),
    (8736.0259610765,70.5093160259),
    (8863.0943847033,70.4246037435),
    (8992.0110611117,70.3386592926),
    (9122.8028737575,70.2514647508),
    (9255.4970971253,70.1630019352),
    (9390.1214024162,70.0732523984),
    (9526.7038633184,69.9834690370),
    (9665.2729618610,69.8976881665),
    (9805.8575943538,69.8106595844),
    (9948.4870774134,69.7223651426),
    (10093.1911540760,69.6327864284),
    (10240.0000000005,69.5419047619),
    (10388.9442297607,69.4497011911),
    (10540.0549032298,69.3561564885),
    (10693.3635320572,69.2612511468),
    (10848.9020862397,69.1649653752),
    (11006.7030007884,69.0672790948),
    (11166.7991824927,68.9681719346),
    (11329.2240167822,68.8676232277),
    (11494.0113746885,68.7656120061),
    (11661.1956199091,68.6621169972),
    (11830.8116159726,68.5571166187),
    (12002.8947335097,68.4505889745),
    (12177.4808576284,68.3425118500),
    (12354.6063953979,68.2328627076),
    (12534.3082834402,68.1216186817),
    (12716.6239956331,68.0087565741),
    (12901.5915509241,67.8942528494),
    (13089.2495212594,67.7780836297),
    (13279.6370396270,67.6602246898),
    (13472.7938082171,67.5406514521),
    (13668.7601067018,67.4193389816),
    (13867.5768006344,67.2962619806),
    (14069.2853499715,67.1713947834),
    (14273.9278177184,67.0447113509),
    (14481.5468787012,66.9161852656),
    (14692.1858284653,66.7857897252),
    (14905.8885923047,66.6534975381),
    (15122.6997344212,66.5192811168),
    (15342.6644672179,66.3831124727),
    (15565.8286607277,66.2449632100),
    (15792.2388521781,66.1048045201),
    (16021.9422556964,65.9626071750),
    (16254.9867721551,65.8183415220),
    (16491.4209991611,65.6719774767),
    (16731.2942411897,65.5234845174),
    (16974.6565198660,65.3728316782),
    (17221.5585843969,65.2199875430),
    (17472.0519221531,65.0649202387),
    (17726.1887694067,64.9075974285),
    (17984.0221222235,64.7479863053),
    (18245.6057475150,64.5860535849),
    (18510.9941942506,64.4217654988),
    (18780.2428048326,64.2550877875),
    (19053.4077266369,64.0859856930),
    (19330.5459237220,63.9144239520),
    (19611.7151887077,63.7403667879),
    (19896.9741548268,63.5637779042),
    (19330.5459237220,63.9144239520),
    (19611.7151887077,63.7403667879),
    (19896.9741548268,63.5637779042),
]
def builtin_target_pts():
    arr = np.array(_BUILTIN_TARGET, dtype=float)
    arr = arr[arr[:, 0].argsort()]
    _, unique_idx = np.unique(arr[:, 0], return_index=True)
    return arr[unique_idx]


def parse_txt(path):
    pts = []
    with open(path, encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    delimiter = ","
    db_col    = 1
    for line in lines:
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        tp = s.split("\t")
        if len(tp) >= 2:
            try:
                float(tp[0]); float(tp[1])
                delimiter = "\t"; db_col = 1
                break
            except ValueError:
                pass
        cp = s.split(",")
        if len(cp) >= 2:
            try:
                float(cp[0])
                delimiter = ","; db_col = 1
                break
            except ValueError:
                header_lower = s.lower()
                if "raw" in header_lower:
                    cols = [c.strip().lower() for c in cp]
                    db_col = cols.index("raw") if "raw" in cols else 1
                continue

    for line in lines:
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        parts = s.split(delimiter)
        if len(parts) <= db_col:
            continue
        try:
            freq = float(parts[0].strip().split()[-1])
            db   = float(parts[db_col].strip().split()[0])
            pts.append((freq, db))
        except (ValueError, IndexError):
            continue

    pts.sort(key=lambda x: x[0])
    if len(pts) == 0:
        raise ValueError(f"No valid data found in: {path}")

    arr = np.array(pts)

    _, unique_idx = np.unique(arr[:, 0], return_index=True)
    arr = arr[unique_idx]

    bad_db = (arr[:, 1] < -120) | (arr[:, 1] > 120)
    if bad_db.any():
        n_bad = bad_db.sum()
        arr   = arr[~bad_db]
        print(f"  WARNING: {n_bad} point(s) with dB outside [-120, 120] removed from {path}")

    bad_f = (arr[:, 0] < 1.0) | (arr[:, 0] > 100000.0)
    if bad_f.any():
        arr = arr[~bad_f]

    if len(arr) < 5:
        raise ValueError(f"Too few valid data points in: {path} (got {len(arr)})")

    return arr


def make_interp(pts):
    return interpolate.interp1d(
        pts[:, 0], pts[:, 1],
        kind="linear", bounds_error=False, fill_value="extrapolate"
    )


def compute_norm_offset(l_pts, r_fn, tgt_fn, norm_freq=1000.0,
                        norm_lo=500.0, norm_hi=2000.0):
    freqs   = l_pts[:, 0]
    avg_db  = (l_pts[:, 1] + r_fn(freqs)) / 2
    raw_corr = tgt_fn(freqs) - avg_db

    mask = (freqs >= norm_lo) & (freqs <= norm_hi)
    if mask.sum() < 2:
        avg_fn = interpolate.interp1d(freqs, avg_db,
                     bounds_error=False, fill_value="extrapolate")
        return float(tgt_fn(norm_freq)) - float(avg_fn(norm_freq))

    log_f_w   = np.log2(freqs[mask])
    log_f_c   = np.log2(norm_freq)
    weights   = np.exp(-0.5 * ((log_f_w - log_f_c) / 0.3) ** 2)
    weights  /= weights.sum() + 1e-12
    return float(np.sum(raw_corr[mask] * weights))


def biquad_peak_db(freqs, fc, gain_db, Q, fs=48000.0):
    A     = 10 ** (gain_db / 40.0)
    w0    = 2 * np.pi * fc / fs
    alpha = np.sin(w0) / (2.0 * Q)
    b0 =  1 + alpha * A;  b1 = -2 * np.cos(w0);  b2 =  1 - alpha * A
    a0 =  1 + alpha / A;  a1 = -2 * np.cos(w0);  a2 =  1 - alpha / A
    b0, b1, b2 = b0/a0, b1/a0, b2/a0
    a1, a2     = a1/a0, a2/a0
    w    = 2 * np.pi * np.asarray(freqs, dtype=float) / fs
    ejw  = np.exp(-1j * w)
    ejw2 = ejw * ejw
    H = (b0 + b1*ejw + b2*ejw2) / (1 + a1*ejw + a2*ejw2)
    return 20 * np.log10(np.maximum(np.abs(H), 1e-10))


def biquad_response_and_grad(f_grid, fcs, gains, Qs, fs=48000.0):
    N = len(fcs); M = len(f_grid)
    LOG10 = np.log(10.0); kappa = LOG10 / 40.0

    fcs_  = fcs[:, None]; gains_ = gains[:, None]; Qs_ = Qs[:, None]
    A_    = 10 ** (gains_ / 40.0)
    w0    = 2 * np.pi * fcs_ / fs
    sin0  = np.sin(w0); cos0 = np.cos(w0)
    alpha = sin0 / (2 * Qs_)
    p = alpha * A_; q = alpha / A_
    a0_sq = (1 + q) ** 2

    b0n = (1 + p) / (1 + q)
    b1n = -2 * cos0 / (1 + q)
    b2n = (1 - p) / (1 + q)
    a2n = (1 - q) / (1 + q)

    w    = 2 * np.pi * f_grid[None, :] / fs
    z1   = np.exp(-1j * w); z2 = z1 * z1

    Bz   = b0n + b1n * z1 + b2n * z2
    Az   = 1   + b1n * z1 + a2n * z2
    H    = Bz / Az
    absH2 = np.maximum(np.abs(H) ** 2, 1e-20)
    resp   = 10 * np.log10(absH2)
    summed = resp.sum(axis=0)

    def _chain(dBz, dAz):
        return (20 / LOG10) * np.real(np.conj(H) * (dBz - H * dAz) / Az) / absH2

    db0n_dg = kappa * (p + q + 2*p*q) / a0_sq
    db1n_dg = -2 * cos0 * q * kappa   / a0_sq
    db2n_dg = kappa * (q - p - 2*p*q) / a0_sq
    da2n_dg =  2 * q * kappa          / a0_sq
    dR_dg   = _chain(db0n_dg + db1n_dg*z1 + db2n_dg*z2,
                     db1n_dg*z1 + da2n_dg*z2)

    dw0_dfc    = 2 * np.pi / fs
    dalpha_dfc = cos0 / (2 * Qs_) * dw0_dfc
    dp_dfc     = dalpha_dfc * A_; dq_dfc = dalpha_dfc / A_
    db1raw_dfc = 2 * sin0 * dw0_dfc
    db0n_dfc = (dp_dfc*(1+q) - (1+p)*dq_dfc) / a0_sq
    db1n_dfc = (db1raw_dfc*(1+q) + 2*cos0*dq_dfc) / a0_sq
    db2n_dfc = (-dp_dfc*(1+q) - (1-p)*dq_dfc) / a0_sq
    da2n_dfc = -2 * dq_dfc / a0_sq
    dR_dlfc  = fcs_ * _chain(db0n_dfc + db1n_dfc*z1 + db2n_dfc*z2,
                              db1n_dfc*z1 + da2n_dfc*z2)

    db0n_dQ = (q - p)   / (Qs_ * a0_sq)
    db1n_dQ = -2*cos0*q / (Qs_ * a0_sq)
    db2n_dQ = (p + q)   / (Qs_ * a0_sq)
    da2n_dQ =  2 * q    / (Qs_ * a0_sq)
    dR_dlq  = Qs_ * _chain(db0n_dQ + db1n_dQ*z1 + db2n_dQ*z2,
                            db1n_dQ*z1 + da2n_dQ*z2)

    return resp, summed, dR_dg, dR_dlfc, dR_dlq


def adaptive_q(fc):
    if   fc <    80: return 0.50
    elif fc <   200: return 0.70
    elif fc <   400: return 1.00
    elif fc <  1000: return 1.50
    elif fc <  2000: return 2.00
    elif fc <  4000: return 2.80
    elif fc <  8000: return 5.00
    elif fc < 12000: return 7.00
    elif fc < 16000: return 10.00
    else:            return 12.00


def make_bands(freq_start, freq_end, n_bands=64, warp=True,
               presence_lo=2000.0, presence_hi=8000.0, presence_boost=2.2,
               bass_lo=20.0, bass_hi=200.0, bass_boost=0.8,
               treble_lo=8000.0, treble_hi=20000.0, treble_boost=1.8):
    if not warp:
        fcs = np.logspace(np.log10(freq_start), np.log10(freq_end), n_bands)
        return fcs, np.array([adaptive_q(fc) for fc in fcs])

    log_s   = np.log10(freq_start); log_e = np.log10(freq_end)

    log_blo  = np.log10(max(bass_lo,     freq_start * 1.001))
    log_bhi  = np.log10(min(bass_hi,     freq_end   * 0.999))
    log_plo  = np.log10(max(presence_lo, freq_start * 1.001))
    log_phi  = np.log10(min(presence_hi, freq_end   * 0.999))
    log_tlo  = np.log10(max(treble_lo,   freq_start * 1.001))
    log_thi  = np.log10(min(treble_hi,   freq_end   * 0.999))

    def seg(lo, hi): return max(hi - lo, 0.0)

    s_bass  = seg(log_blo, log_bhi)
    s_lomid = seg(log_bhi, log_plo)
    s_pres  = seg(log_plo, log_phi)
    s_treb  = seg(log_tlo, log_thi)
    s_rest  = max((log_e - log_s)
                  - s_bass - s_lomid - s_pres - s_treb, 0.0)

    total = (bass_boost * s_bass + s_lomid +
             presence_boost * s_pres + treble_boost * s_treb + s_rest)

    if total <= 0:
        fcs = np.logspace(log_s, log_e, n_bands)
        return fcs, np.array([adaptive_q(fc) for fc in fcs])

    c_bass  = bass_boost     * s_bass  / total
    c_lomid = c_bass  + s_lomid        / total
    c_pres  = c_lomid + presence_boost * s_pres  / total
    c_treb  = c_pres  + treble_boost   * s_treb  / total

    u = np.linspace(0.0, 1.0, n_bands)
    log_fcs = np.zeros(n_bands)

    for i, ui in enumerate(u):
        if ui <= c_bass and s_bass > 0:
            t = ui / c_bass
            log_fcs[i] = log_blo + t * s_bass
        elif ui <= c_lomid and s_lomid > 0:
            t = (ui - c_bass) / (c_lomid - c_bass)
            log_fcs[i] = log_bhi + t * s_lomid
        elif ui <= c_pres and s_pres > 0:
            t = (ui - c_lomid) / (c_pres - c_lomid)
            log_fcs[i] = log_plo + t * s_pres
        elif ui <= c_treb and s_treb > 0:
            t = (ui - c_pres) / (c_treb - c_pres)
            log_fcs[i] = log_tlo + t * s_treb
        else:
            denom = max(1.0 - c_treb, 1e-9)
            t = (ui - c_treb) / denom
            log_fcs[i] = log_thi + t * (log_e - log_thi)

    fcs = np.clip(10 ** log_fcs, freq_start, freq_end)
    return fcs, np.array([adaptive_q(fc) for fc in fcs])


def perceptual_weights(f_fit, peak=2.5):
    log_f = np.log10(f_fit); dlog = np.gradient(log_f)
    base  = 1.0 / (np.abs(dlog) + 1e-12)
    perc  = np.ones(len(f_fit))
    log_200 = np.log10(200.0); log_4k  = np.log10(4000.0)
    log_12k = np.log10(12000.0); log_20k = np.log10(20000.0)
    for i, lf in enumerate(log_f):
        if   lf <= log_200: perc[i] = 1.0
        elif lf <= log_4k:
            t = (lf - log_200) / (log_4k - log_200)
            perc[i] = 1.0 + (peak - 1.0) * t
        elif lf <= log_12k: perc[i] = peak
        else:
            t = (lf - log_12k) / max(log_20k - log_12k, 1e-9)
            perc[i] = peak - (peak - 1.0) * min(t, 1.0)
    return base * perc


def minimum_phase_correction(c_fit, f_fit, f_start=20.0, f_end=20000.0,
                             fs=48000.0):
    N_fft = 2048
    f_nyq = fs / 2.0

    f_lin = np.linspace(0, f_nyq, N_fft // 2 + 1)  # 0 to fs/2

    c_lin = np.interp(f_lin,
                      np.clip(f_fit, f_start, f_end),
                      c_fit,
                      left=c_fit[0], right=c_fit[-1])

    _LN10_OVER_20 = np.log(10.0) / 20.0
    c_lin_nats = c_lin * _LN10_OVER_20   # dB → natural log amplitude

    C_full = np.zeros(N_fft)
    C_full[:N_fft//2+1] = c_lin_nats
    C_full[N_fft//2+1:] = c_lin_nats[-2:0:-1]  # mirror for real

    cepstrum = np.fft.ifft(C_full).real

    cepstrum_mp = cepstrum.copy()
    half = N_fft // 2
    cepstrum_mp[half+1:]  = 0.0   # zero negative quefrency
    cepstrum_mp[1:half]  *= 2.0   # double positive (energy conservation)
    cepstrum_mp[0]       *= 1.0   # DC stays

    C_mp = np.fft.fft(cepstrum_mp).real
    c_lin_mp_nats = C_mp[:N_fft//2+1]

    c_lin_mp = c_lin_mp_nats / _LN10_OVER_20

    c_fit_mp = np.interp(f_fit, f_lin, c_lin_mp)


    def erb_hz(f):
        return 24.7 * (4.37 * np.asarray(f) / 1000.0 + 1.0)

    def erb_normalized(f):
        return erb_hz(f) / erb_hz(1000.0)

    erb_n     = erb_normalized(f_fit)
    erb_pivot = erb_normalized(3000.0)
    k_erb     = 4.0


    from scipy import ndimage as _ndimage
    from scipy.interpolate import UnivariateSpline

    log_f_fit   = np.log2(np.maximum(f_fit, 1.0))

    dlog_mean = float(np.mean(np.diff(log_f_fit)))
    sg_sigma  = max(0.1 / dlog_mean, 1.0)
    c_smooth_for_curv = ndimage.gaussian_filter1d(c_fit, sigma=sg_sigma)

    n_oct  = float(np.log2(f_fit[-1] / max(f_fit[0], 1.0)))
    s_val  = max(n_oct * 0.4, 1.0)   # ~0.4 dB² per octave tolerance
    try:
        spl = UnivariateSpline(log_f_fit, c_smooth_for_curv,
                               k=4, s=s_val, ext=3)
        spl_d2 = spl.derivative(n=2)
        abs_curv = np.abs(spl_d2(log_f_fit))
    except Exception:
        dc_dlogf   = np.gradient(c_smooth_for_curv, log_f_fit)
        abs_curv   = np.abs(np.gradient(dc_dlogf, log_f_fit))

    curv_n      = abs_curv / (abs_curv + 2.0)   # range [0,1), ~0.5 at 2 dB/oct²

    raw_blend   = 0.40 + 0.50 / (1.0 + np.exp(
        k_erb * (erb_n - erb_pivot) - 2.0 * curv_n
    ))

    sub_bass_boost = 0.85 * np.exp(-((f_fit - 60.0) ** 2) / (2 * 80.0 ** 2))
    mp_blend = np.maximum(raw_blend, sub_bass_boost)
    mp_blend = np.clip(mp_blend, 0.40, 0.92)

    return mp_blend * c_fit_mp + (1.0 - mp_blend) * c_fit


def a_weighting_db(freqs):
    f = np.asarray(freqs, dtype=float)
    f2 = f ** 2
    num   = 12194.0**2 * f2**2
    denom = ((f2 + 20.6**2) *
             np.sqrt((f2 + 107.7**2) * (f2 + 737.9**2)) *
             (f2 + 12194.0**2))
    Ra = num / (denom + 1e-300)
    Ra_1k = (12194.0**2 * 1000.0**4) / (
        (1000.0**2 + 20.6**2) *
        np.sqrt((1000.0**2 + 107.7**2) * (1000.0**2 + 737.9**2)) *
        (1000.0**2 + 12194.0**2))
    return 20.0 * np.log10(Ra / Ra_1k + 1e-300)


def masking_weights(f_fit):
    aw = a_weighting_db(f_fit)
    aw_shifted = aw - aw.min()
    w = 10 ** (aw_shifted / 20.0)
    return w / (w.mean() + 1e-12)


def perceptual_pole_weight(fc):
    if   fc <   150: return 0.3
    elif fc <   500: return 0.6
    elif fc <  2000: return 1.2
    elif fc <  5000: return 3.0   # presence: most critical
    elif fc <  8000: return 2.0
    elif fc < 12000: return 1.0
    else:            return 0.5


def biquad_pole_radius(fc, Q, fs=48000.0):
    w0    = 2 * np.pi * fc / fs
    alpha = np.sin(w0) / (2.0 * Q)
    a0 = 1.0 + alpha
    a2 = (1.0 - alpha) / a0
    r  = np.sqrt(max(abs(a2), 1e-10))
    return min(r, 0.9999)  # cap at 0.9999 to avoid div-by-zero


def pole_radius_penalty(fcs, gains, Qs, lam_gd, fs=48000.0, hf_gamma=0.7):
    f_nyq  = fs / 2.0
    fcs_a  = np.asarray(fcs, dtype=float)
    g_a    = np.asarray(gains, dtype=float)
    Q_a    = np.asarray(Qs, dtype=float)
    w0     = 2 * np.pi * fcs_a / fs
    alpha  = np.sin(w0) / (2.0 * Q_a)
    a2     = (1.0 - alpha) / (1.0 + alpha)
    r      = np.sqrt(np.maximum(np.abs(a2), 1e-10))
    r      = np.minimum(r, 0.9999)
    w_perc = np.array([perceptual_pole_weight(fc) for fc in fcs_a])
    w_hf   = (fcs_a / f_nyq) ** hf_gamma
    pen    = w_perc * w_hf * g_a**2 / np.maximum((1.0 - r)**2, 1e-6)
    return float(lam_gd * pen.sum())


def pole_radius_gradient(fcs, gains, Qs, lam_gd, fs=48000.0, hf_gamma=0.7):
    f_nyq  = fs / 2.0
    fcs_a  = np.asarray(fcs); g_a = np.asarray(gains); Q_a = np.asarray(Qs)
    w0     = 2 * np.pi * fcs_a / fs
    alpha  = np.sin(w0) / (2.0 * Q_a)
    a2     = (1.0 - alpha) / (1.0 + alpha)
    r      = np.sqrt(np.maximum(np.abs(a2), 1e-10))
    r      = np.minimum(r, 0.9999)
    w_perc = np.array([perceptual_pole_weight(fc) for fc in fcs_a])
    w_hf   = (fcs_a / f_nyq) ** hf_gamma
    return lam_gd * w_perc * w_hf * 2.0 * g_a / np.maximum((1.0 - r) ** 2, 1e-6)


def biquad_complex_response(freqs, fc, gain_db, Q, fs=48000.0):
    A     = 10 ** (gain_db / 40.0)
    w0    = 2 * np.pi * fc / fs
    alpha = np.sin(w0) / (2.0 * Q)
    b0 = (1 + alpha * A) / (1 + alpha / A)
    b1 = (-2 * np.cos(w0)) / (1 + alpha / A)
    b2 = (1 - alpha * A) / (1 + alpha / A)
    a1 = (-2 * np.cos(w0)) / (1 + alpha / A)
    a2 = (1 - alpha / A) / (1 + alpha / A)
    w    = 2 * np.pi * np.asarray(freqs) / fs
    z    = np.exp(1j * w)
    num  = b0 + b1 / z + b2 / (z ** 2)
    den  = 1.0 + a1 / z + a2 / (z ** 2)
    return num / den


def phase_slope_penalty(freqs, fcs, gains, Qs, lam_ph, fs=48000.0):
    if lam_ph == 0.0:
        return 0.0, np.zeros(len(gains))

    N  = len(fcs)
    M  = len(freqs)
    log_f = np.log2(np.maximum(freqs, 1.0))

    H_total = np.ones(M, dtype=complex)
    H_each  = []
    for fc, g, Q in zip(fcs, gains, Qs):
        Hk = biquad_complex_response(freqs, fc, g, Q, fs)
        H_total *= Hk
        H_each.append(Hk)

    phase_total = np.unwrap(np.angle(H_total))

    dlogf   = np.gradient(log_f)
    dph_dlf = np.gradient(phase_total, log_f)    # Δφ/Δlogf

    w_ph = np.array([perceptual_pole_weight(f) for f in freqs], dtype=float)
    w_ph = w_ph / (w_ph.mean() + 1e-12)

    penalty = float(lam_ph * np.sum(w_ph * dph_dlf ** 2))

    _LN10_OVER_20 = np.log(10.0) / 20.0
    grad = np.zeros(N)
    for k, (fc, g, Q, Hk) in enumerate(zip(fcs, gains, Qs, H_each)):
        eps = 0.01
        Hk_p = biquad_complex_response(freqs, fc, g + eps, Q, fs)
        dHk_dg = (Hk_p - Hk) / eps
        dphi_dg_k = np.imag(dHk_dg / (Hk + 1e-30))
        dphi_slope_dg = np.gradient(dphi_dg_k, log_f)
        grad[k] = float(lam_ph * 2.0 * np.sum(w_ph * dph_dlf * dphi_slope_dg))

    return penalty, grad

def freq_dependent_q_scale(fc):
    if   fc <   200: return 0.2   # sub-bass/bass: wide filters OK
    elif fc <   500: return 0.5   # upper bass
    elif fc <  1000: return 0.8   # low-mid
    elif fc <  2000: return 1.2   # mid
    elif fc <  4000: return 2.5   # presence: most critical ← strict
    elif fc <  6000: return 2.0   # upper presence
    elif fc <  8000: return 1.5   # lower treble
    elif fc < 12000: return 1.0   # upper treble
    else:            return 0.7   # air: moderate


def q_log_penalty(Qs, gains, fcs=None):
    if fcs is None:
        scales = np.ones(len(Qs))
    else:
        scales = np.array([freq_dependent_q_scale(fc) for fc in fcs])
    return float(np.sum(scales * (np.log(np.maximum(Qs, 0.1)) ** 2) * np.abs(gains)))


def q_penalty_gradient(Qs, gains, fcs=None):
    if fcs is None:
        scales = np.ones(len(Qs))
    else:
        scales = np.array([freq_dependent_q_scale(fc) for fc in fcs])
    sign_g = np.sign(gains)
    return scales * (np.log(np.maximum(Qs, 0.1)) ** 2) * sign_g


def estimate_hf_rolloff_slope(f_fit, raw_avg_db, f_lo=6000.0, f_hi=14000.0):
    mask = (f_fit >= f_lo) & (f_fit <= f_hi)
    if mask.sum() < 3:
        return 0.0
    log_f  = np.log2(f_fit[mask])
    r_seg  = raw_avg_db[mask]
    coeffs = np.polyfit(log_f, r_seg, 1)
    return float(coeffs[0])


def driver_capability_penalty(fcs, gains, hf_threshold=8000.0,
                              hf_max_boost=6.0, rolloff_slope=0.0):
    slope_penalty = min(rolloff_slope * 0.5, 0.0)  # negative if steep rolloff

    fcs_a = np.asarray(fcs, dtype=float)
    g_a   = np.asarray(gains, dtype=float)

    caps = np.where(fcs_a < 15000,
                    max(hf_max_boost + slope_penalty, 2.0),
           np.where(fcs_a < 17000,
                    max(2.5 + slope_penalty, 1.0),
           np.where(fcs_a < 19000,
                    max(1.5 + slope_penalty, 0.5),
                    max(0.8 + slope_penalty, 0.3))))

    excess = np.maximum(g_a - caps, 0.0)
    return float((excess ** 2).sum())


def per_band_lambda(fcs, lam_base):
    lam = np.ones(len(fcs)) * lam_base
    for k, fc in enumerate(fcs):
        if fc > 2000:
            t = (np.log10(fc) - np.log10(2000)) / (np.log10(20000) - np.log10(2000))
            lam[k] = lam_base * (1.0 + 5.0 * t)
    return lam


def make_third_deriv_matrix(N, lam_td):
    if N < 4:
        return np.zeros((N, N))
    rows = N - 3
    D3 = np.zeros((rows, N))
    coeffs = np.array([1.0, -3.0, 3.0, -1.0])
    for i in range(rows):
        D3[i, i:i+4] = coeffs
    M_td = D3.T @ D3
    diag_vals = np.diag(M_td)
    mean_diag = diag_vals[diag_vals > 0].mean() if (diag_vals > 0).any() else 1.0
    M_td /= mean_diag
    return lam_td * M_td


def make_smooth_matrix_freq(fcs, lam_smooth_base, lam_third_deriv=None):
    N = len(fcs); s = np.ones(N - 1) * lam_smooth_base
    for k in range(N - 1):
        fc_mid = np.sqrt(fcs[k] * fcs[k + 1])
        if   fc_mid <   200: s[k] = lam_smooth_base * 2.5
        elif fc_mid <   800: s[k] = lam_smooth_base * 1.5
        elif fc_mid <  8000: s[k] = lam_smooth_base * 0.4
        elif fc_mid < 12000: s[k] = lam_smooth_base * 0.8
        else:                s[k] = lam_smooth_base * 1.5
    M = np.zeros((N, N))
    for k in range(N - 1):
        M[k,   k  ] += s[k]; M[k+1, k+1] += s[k]
        M[k,   k+1] -= s[k]; M[k+1, k  ] -= s[k]

    lam_td = lam_third_deriv if lam_third_deriv is not None else lam_smooth_base * 0.3
    M += make_third_deriv_matrix(N, lam_td)
    return M


def make_energy_matrix(fcs, Qs, lam_energy):
    bw   = fcs / Qs
    bw_n = bw / (np.median(bw) + 1e-12)
    return np.diag(lam_energy * bw_n)


_LN10_OVER_20 = np.log(10.0) / 20.0

def cosh_weights(c_db):
    return np.cosh(c_db * _LN10_OVER_20)


def smooth_residual_logfreq(residual, f_fit, sigma_oct=0.167):
    log_f  = np.log2(f_fit); dlog = np.mean(np.diff(log_f))
    if dlog <= 0: return residual.copy()
    sigma_pts = sigma_oct / dlog
    return ndimage.gaussian_filter1d(residual, sigma=max(sigma_pts, 0.5))


def freq_jitter_smooth(c_fit, f_fit, sigma_oct=0.08):
    log_f = np.log2(np.maximum(f_fit, 1.0))
    dlog  = np.mean(np.diff(log_f))
    if dlog <= 0:
        return c_fit.copy()

    log_1k   = np.log2(1000.0)
    sigma_f  = 0.055 + 0.028 * (log_f - log_1k) ** 2
    sigma_f  = np.clip(sigma_f, 0.045, 0.14)

    s_fine   = max(0.045 / dlog, 0.3)
    s_med    = max(0.080 / dlog, 0.3)
    s_coarse = max(0.130 / dlog, 0.3)

    c_fine   = ndimage.gaussian_filter1d(c_fit, sigma=s_fine)
    c_med    = ndimage.gaussian_filter1d(c_fit, sigma=s_med)
    c_coarse = ndimage.gaussian_filter1d(c_fit, sigma=s_coarse)

    s_vals = np.array([0.045, 0.080, 0.130])
    w = np.zeros((3, len(f_fit)))
    for i in range(len(f_fit)):
        sv = sigma_f[i]
        if sv <= s_vals[0]:
            w[0, i] = 1.0
        elif sv <= s_vals[1]:
            t = (sv - s_vals[0]) / (s_vals[1] - s_vals[0])
            w[0, i] = 1.0 - t; w[1, i] = t
        elif sv <= s_vals[2]:
            t = (sv - s_vals[1]) / (s_vals[2] - s_vals[1])
            w[1, i] = 1.0 - t; w[2, i] = t
        else:
            w[2, i] = 1.0

    return w[0] * c_fine + w[1] * c_med + w[2] * c_coarse


def multiresolution_irls_weights(residual, f_fit, base_weights,
                                  alpha=2.0, beta=1.2, gamma=0.15,
                                  sigma_macro=1.0, sigma_meso=0.25,
                                  cap=3.0):
    log_f = np.log2(f_fit)
    dlog  = np.mean(np.diff(log_f))
    if dlog <= 0:
        return base_weights.copy()

    abs_res = np.abs(residual)

    log_500  = np.log2(500.0);  log_2k  = np.log2(2000.0)
    log_6k   = np.log2(6000.0); log_10k = np.log2(10000.0)

    def soft_mask(log_f_arr, lo, hi, sharpness=3.0):
        rise = 1.0 / (1.0 + np.exp(-sharpness * (log_f_arr - lo)))
        fall = 1.0 / (1.0 + np.exp(-sharpness * (hi - log_f_arr)))
        return rise * fall

    mask_bass     = soft_mask(log_f, -np.inf, log_500,  sharpness=4.0)
    mask_presence = soft_mask(log_f, log_2k,  log_6k,   sharpness=4.0)
    mask_air      = soft_mask(log_f, log_10k, np.inf,   sharpness=4.0)
    mask_mid      = np.clip(1.0 - mask_bass - mask_presence - mask_air, 0.0, 1.0)

    def _smooth(arr, sigma_oct):
        sigma_pts = max(sigma_oct / dlog, 0.3)
        return ndimage.gaussian_filter1d(arr, sigma=sigma_pts)

    mac_bass     = _smooth(abs_res, sigma_macro * 1.5)
    mac_mid      = _smooth(abs_res, sigma_macro * 1.0)
    mac_presence = _smooth(abs_res, sigma_macro * 1.0)
    mac_air      = _smooth(abs_res, sigma_macro * 0.7)
    res_macro = (mask_bass     * mac_bass     +
                 mask_mid      * mac_mid      +
                 mask_presence * mac_presence +
                 mask_air      * mac_air)

    abs_res_nomacro = np.maximum(abs_res - res_macro, 0.0)
    mes_bass     = _smooth(abs_res_nomacro, sigma_meso * 1.8)
    mes_mid      = _smooth(abs_res_nomacro, sigma_meso * 1.0)
    mes_presence = _smooth(abs_res_nomacro, sigma_meso * 0.6)
    mes_air      = _smooth(abs_res_nomacro, sigma_meso * 0.4)
    res_meso = (mask_bass     * mes_bass     +
                mask_mid      * mes_mid      +
                mask_presence * mes_presence +
                mask_air      * mes_air)

    res_micro = np.maximum(abs_res - res_macro - res_meso, 0.0)

    extra = 1.0 + alpha * res_macro + beta * res_meso + gamma * res_micro
    extra = np.clip(extra, 1.0, cap)

    W2 = base_weights * extra
    W2 = W2 / (W2.sum() + 1e-12) * len(W2)
    return W2


def compute_adaptive_fc_bounds(fcs, f_fit, residual, fc_range_base,
                               min_range=0.08, max_range=0.35):
    N = len(fcs); fc_ranges = np.full(N, fc_range_base)
    log_f_fit = np.log2(f_fit); log_fcs = np.log2(fcs)
    abs_res   = np.abs(residual)
    is_peak   = np.zeros(len(f_fit), dtype=bool)
    is_peak[1:-1] = (abs_res[1:-1] > abs_res[:-2]) & (abs_res[1:-1] > abs_res[2:])
    peak_idx  = np.where(is_peak)[0]
    if len(peak_idx) == 0: return fc_ranges

    for k in range(N):
        dist       = np.abs(log_f_fit[peak_idx] - log_fcs[k])
        nearest_pk = peak_idx[np.argmin(dist)]
        pk_val     = abs_res[nearest_pk]
        if pk_val < 0.3:
            fc_ranges[k] = fc_range_base; continue
        half_val = pk_val * 0.707
        left = nearest_pk
        while left > 0 and abs_res[left - 1] > half_val: left -= 1
        right = nearest_pk
        while right < len(f_fit) - 1 and abs_res[right + 1] > half_val: right += 1
        width_oct    = log_f_fit[right] - log_f_fit[left]
        t            = np.clip((width_oct - 0.25) / 0.75, 0.0, 1.0)
        fc_ranges[k] = min_range + t * (max_range - min_range)
    return fc_ranges


def optimize_gains_biquad(fcs, Qs, f_fit, c_fit, weights, lam_base,
                          M_smooth, max_iter, lam_energy=0.05, fs=48000.0,
                          lam_q=0.02, lam_gd=0.001, lam_driver=0.5,
                          use_minphase=True, raw_avg_db=None, lam_ph=0.0005):
    N      = len(fcs)
    Qs_arr = np.array([adaptive_q(fc) for fc in fcs]) if Qs is None else np.array(Qs)
    lam_diag = np.diag(per_band_lambda(fcs, lam_base))
    E_diag   = make_energy_matrix(fcs, Qs_arr, lam_energy)
    reg      = lam_diag + M_smooth + E_diag

    if use_minphase:
        c_fit_opt = minimum_phase_correction(c_fit, f_fit, fs=fs)
    else:
        c_fit_opt = c_fit

    c_fit_opt = freq_jitter_smooth(c_fit_opt, f_fit, sigma_oct=0.08)
    if raw_avg_db is not None:
        raw_fit = raw_avg_db  # already on same grid — no-op interp removed
        rolloff_slope = estimate_hf_rolloff_slope(f_fit, raw_fit)
    else:
        rolloff_slope = 0.0

    s_cosh = cosh_weights(c_fit_opt)
    s_aw   = masking_weights(f_fit)
    W_perc = weights / weights.sum() * len(weights)
    W      = W_perc * s_cosh * s_aw
    W      = W / W.sum() * len(W)

    W = W / (np.median(W) + 1e-12)
    W = np.clip(W, 0.25, 6.0)
    W = W / W.sum() * len(W)   # renormalize after clamp

    g_zero = np.zeros(N)
    _, _, dR_dg0, _, _ = biquad_response_and_grad(f_fit, fcs, g_zero, Qs_arr, fs)
    J0   = dR_dg0.T                              # (M, N)
    WJ   = (J0.T * W).T                          # (M, N) weighted

    col_rms = np.sqrt(np.mean(WJ ** 2, axis=0))  # (N,)
    col_rms = np.maximum(col_rms, 1e-8)           # avoid div-by-zero
    J0_n    = J0   / col_rms                      # normalized Jacobian
    WJ_n    = WJ   / col_rms                      # normalized weighted J

    JtWJ = WJ_n.T @ J0_n + reg / np.outer(col_rms, col_rms)
    JtWc = WJ_n.T @ c_fit_opt
    try:
        g0_scaled = np.linalg.solve(JtWJ, JtWc)
        g0 = np.clip(g0_scaled / col_rms, -12, 12)   # undo scaling
    except np.linalg.LinAlgError:
        g0 = np.zeros(N)

    def freq_max_boost(fc):
        if   fc <  120: return 12.0   # sub-bass: excursion-limited but high tolerance
        elif fc <  500: return 9.0    # bass
        elif fc < 2000: return 7.0    # mid
        elif fc < 8000: return 5.0    # presence: tightest
        else:           return dynamic_max  # air: existing dynamic constraint

    dynamic_max = max(6.0 + min(rolloff_slope * 0.5, 0.0), 2.0)

    M = len(f_fit)   # needed by phase slope penalty inside closure
    def make_obj_grad(W_):
        def obj_and_grad(g):
            _, summed, dR_dg, _, _ = biquad_response_and_grad(
                f_fit, fcs, g, Qs_arr, fs)
            r       = summed - c_fit_opt

            _delta = 1.0
            abs_r   = np.abs(r)
            huber_v = np.where(abs_r <= _delta,
                               0.5 * r ** 2,
                               _delta * (abs_r - 0.5 * _delta))
            huber_g = np.where(abs_r <= _delta, r, _delta * np.sign(r))
            fit_val = float(np.dot(W_, huber_v))
            wr      = W_ * huber_g          # effective residual for gradient
            reg_val = float(g @ reg @ g)

            pr_pen  = pole_radius_penalty(fcs, g, Qs_arr, lam_gd, fs)

            q_pen   = lam_q * q_log_penalty(Qs_arr, g, fcs)

            f_ph_idx = np.round(np.linspace(0, M-1, min(60, M))).astype(int)
            f_ph_grid = f_fit[f_ph_idx]
            ph_pen, grad_ph_sparse = phase_slope_penalty(
                f_ph_grid, fcs, g, Qs_arr, lam_ph, fs)

            _k_logistic = 3.0
            drv_pen = 0.0
            for fc, gk in zip(fcs, g):
                cap = freq_max_boost(fc)
                x = _k_logistic * (gk - cap)
                drv_pen += lam_driver * np.log1p(np.exp(np.clip(x, -30, 30))) / _k_logistic

            total    = fit_val + reg_val + pr_pen + q_pen + ph_pen + drv_pen

            grad_fit = dR_dg @ wr           # note: factor 2 absorbed into huber_g
            grad_reg = 2.0 * reg @ g
            grad_pr  = pole_radius_gradient(fcs, g, Qs_arr, lam_gd, fs)
            grad_q   = lam_q * q_penalty_gradient(Qs_arr, g, fcs)
            grad_drv = np.zeros(N)
            for k, (fc, gk) in enumerate(zip(fcs, g)):
                cap = freq_max_boost(fc)
                x = _k_logistic * (gk - cap)
                sigmoid = 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))
                grad_drv[k] = lam_driver * sigmoid

            return total, grad_fit + grad_reg + grad_pr + grad_q + grad_ph_sparse + grad_drv
        return obj_and_grad

    def safe_q_cap(fc, fs_=fs):
        if   fc <   200: max_r = 0.9999   # sub-bass/bass: nearly unconstrained, penalty handles it
        elif fc <   500: max_r = 0.9995   # upper bass
        elif fc <  1000: max_r = 0.9990   # low-mid
        elif fc <  2000: max_r = 0.9970   # mid
        elif fc <  4000: max_r = 0.9960   # presence: strict
        elif fc <  8000: max_r = 0.9950   # treble
        else:            max_r = 0.9940   # air: strictest
        lo_, hi_ = 0.1, 300.0
        for _ in range(40):
            mid = (lo_ + hi_) / 2
            w0_    = 2 * np.pi * fc / fs_
            alpha_ = np.sin(w0_) / (2.0 * mid)
            a2_    = (1.0 - alpha_) / (1.0 + alpha_)
            r_     = np.sqrt(abs(a2_))
            if r_ < max_r: lo_ = mid
            else:          hi_ = mid
        return lo_

    q_caps  = np.array([safe_q_cap(fc) for fc in fcs])
    Qs_arr  = np.clip(Qs_arr, 0.1, q_caps)

    r1 = optimize.minimize(
        make_obj_grad(W), g0, jac=True,
        bounds=[(-12, 12)] * N, method="L-BFGS-B",
        options={"maxiter": max_iter // 2, "ftol": 1e-15, "gtol": 1e-11}
    )
    g1 = r1.x

    _, summed1, _, _, _ = biquad_response_and_grad(f_fit, fcs, g1, Qs_arr, fs)
    res1_signed = summed1 - c_fit_opt
    W2 = multiresolution_irls_weights(res1_signed, f_fit, W)

    r2 = optimize.minimize(
        make_obj_grad(W2), g1, jac=True,
        bounds=[(-12, 12)] * N, method="L-BFGS-B",
        options={"maxiter": max_iter, "ftol": 1e-15, "gtol": 1e-11}
    )
    return r2.x, (r1.success or r2.success)


def optimize_joint(gains0, fcs_nom, Qs_nom, f_fit, c_fit, weights,
                   lam_base, M_smooth, max_iter, fs=48000.0,
                   fc_range=0.15, q_lo=0.7, q_hi=12.0, n_grid=500,
                   lam_energy=0.05):
    if max_iter == 0:
        return gains0.copy(), fcs_nom.copy(), Qs_nom.copy(), True

    N = len(fcs_nom)
    n_j = min(n_grid, len(f_fit))

    _, summed0_pre, _, _, _ = biquad_response_and_grad(f_fit, fcs_nom, gains0, Qs_nom, fs)
    res0_pre = np.abs(summed0_pre - c_fit)
    w_perc   = perceptual_weights(f_fit)
    w_samp   = w_perc * (res0_pre ** 0.7 + 1e-6)   # residual-aware blend
    w_samp   = w_samp / w_samp.sum()
    rng      = np.random.default_rng(seed=42)        # deterministic
    idx_j    = np.sort(rng.choice(len(f_fit), size=n_j, replace=False, p=w_samp))
    f_j      = f_fit[idx_j]; c_j = c_fit[idx_j]
    w_j      = perceptual_weights(f_j) * cosh_weights(c_j)
    w_j      = w_j / w_j.sum() * len(w_j)

    lam_diag = np.diag(per_band_lambda(fcs_nom, lam_base))
    E_diag   = make_energy_matrix(fcs_nom, Qs_nom, lam_energy)
    reg      = lam_diag + M_smooth + E_diag

    _, summed0, _, _, _ = biquad_response_and_grad(f_fit, fcs_nom, gains0, Qs_nom, fs)
    res0      = np.abs(summed0 - c_fit)
    fc_ranges = compute_adaptive_fc_bounds(fcs_nom, f_fit, res0, fc_range)

    lfc_lo = np.log(1.0 - fc_ranges); lfc_hi = np.log(1.0 + fc_ranges)
    lq_lo  = np.log(q_lo);            lq_hi  = np.log(q_hi)
    bounds  = ([(-12, 12)] * N +
               [(float(lfc_lo[k]), float(lfc_hi[k])) for k in range(N)] +
               [(lq_lo, lq_hi)] * N)

    x0 = np.concatenate([gains0, np.zeros(N), np.zeros(N)])

    def obj_and_grad(x):
        g   = x[:N]
        fcs = fcs_nom * np.exp(x[N:2*N])
        Qs  = Qs_nom  * np.exp(x[2*N:])

        _, summed, dR_dg, dR_dlfc, dR_dlq = biquad_response_and_grad(f_j, fcs, g, Qs, fs)
        r  = summed - c_j; wr = w_j * r

        total    = float(np.dot(wr, r)) + float(g @ reg @ g)
        grad_g   = 2 * (dR_dg   @ wr) + 2 * reg @ g
        grad_lfc = 2 * (dR_dlfc @ wr)
        grad_lq  = 2 * (dR_dlq  @ wr)
        return total, np.concatenate([grad_g, grad_lfc, grad_lq])

    result = optimize.minimize(
        obj_and_grad, x0, jac=True, bounds=bounds, method="L-BFGS-B",
        options={"maxiter": max_iter, "ftol": 1e-15, "gtol": 1e-10}
    )
    g_out   = np.clip(result.x[:N], -12, 12)
    fcs_out = fcs_nom * np.exp(np.clip(result.x[N:2*N],   lfc_lo, lfc_hi))
    Qs_out  = Qs_nom  * np.exp(np.clip(result.x[2*N:3*N], lq_lo,  lq_hi))
    return g_out, fcs_out, Qs_out, result.success


def _local_maxima(arr, min_val=0.0):
    is_max = (arr[1:-1] > arr[:-2]) & (arr[1:-1] > arr[2:]) & (arr[1:-1] > min_val)
    return np.where(is_max)[0] + 1


def adaptive_band_realloc(f_fit, c_fit, fcs, Qs, gains,
                          waste_thresh=0.5, residual_min=0.5, fs=48000.0):
    _, summed, _, _, _ = biquad_response_and_grad(f_fit, fcs, gains, Qs, fs)
    residual = np.abs(summed - c_fit)

    wasted = np.where(np.abs(gains) < waste_thresh)[0]
    if len(wasted) == 0: return fcs.copy(), Qs.copy(), 0
    wasted = wasted[np.argsort(np.abs(gains[wasted]))]

    peaks = _local_maxima(residual, min_val=residual_min)
    if len(peaks) == 0:
        peaks = np.argsort(residual)[-len(wasted):][::-1]
    else:
        peaks = peaks[np.argsort(-residual[peaks])]

    new_fcs = fcs.copy().tolist(); new_Qs = Qs.copy().tolist(); n_moved = 0
    for peak_idx in peaks:
        if n_moved >= len(wasted): break
        f_hot = f_fit[peak_idx]
        covered = any(
            k not in wasted[n_moved:] and abs(np.log2(f_hot / fc)) < 0.33
            for k, fc in enumerate(new_fcs)
        )
        if covered: continue
        band_idx = wasted[n_moved]
        new_fcs[band_idx] = f_hot; new_Qs[band_idx] = adaptive_q(f_hot)
        n_moved += 1

    fcs_arr = np.array(new_fcs); Qs_arr = np.array(new_Qs)
    sort_idx = np.argsort(fcs_arr)
    return fcs_arr[sort_idx], Qs_arr[sort_idx], n_moved


def iterative_band_spawn(f_fit, c_fit, weights, lam_base, lam_smooth,
                          lam_energy, lam_q, lam_gd, lam_driver,
                          f_start, f_end, use_minphase, raw_avg_db,
                          iters, max_iters_spawn,
                          n_start=32, n_max=64, n_step=8,
                          rmse_plateau=0.005, fs=48000.0):
    use_warp = True
    fcs, Qs = make_bands(f_start, f_end, n_bands=n_start, warp=use_warp)
    M_smooth = make_smooth_matrix_freq(fcs, lam_smooth)
    w        = perceptual_weights(f_fit)

    print(f"  Spawning: start={n_start} bands...")
    gains, _ = optimize_gains_biquad(
        fcs, Qs, f_fit, c_fit, w, lam_base, M_smooth, iters,
        lam_energy=lam_energy, lam_q=lam_q, lam_gd=lam_gd, lam_driver=lam_driver,
        use_minphase=use_minphase, raw_avg_db=raw_avg_db, fs=fs)
    rmse_prev, _, _, _ = compute_stats_biquad(fcs, Qs, gains, c_fit, f_fit, fs)
    print(f"  Spawning: n={n_start} RMSE={rmse_prev:.4f} dB")

    n_cur = n_start
    spawn_iter = 0
    while n_cur < n_max and spawn_iter < max_iters_spawn:
        spawn_iter += 1
        n_spawn = min(n_step, n_max - n_cur)

        _, summed, _, _, _ = biquad_response_and_grad(f_fit, fcs, gains, Qs, fs)
        residual  = np.abs(summed - c_fit)
        peaks     = _local_maxima(residual, min_val=0.3)
        if len(peaks) == 0:
            peaks = np.argsort(residual)[-n_spawn:][::-1]
        else:
            peaks = peaks[np.argsort(-residual[peaks])]

        new_fcs = list(fcs); new_Qs = list(Qs); added = 0
        for pk in peaks:
            if added >= n_spawn: break
            f_hot = f_fit[pk]
            already = any(abs(np.log2(f_hot / fc)) < 0.25 for fc in new_fcs)
            if already: continue
            new_fcs.append(f_hot); new_Qs.append(adaptive_q(f_hot))
            added += 1

        if added == 0:
            print(f"  Spawning: no new hotspot candidates, stopping.")
            break

        fcs_new = np.array(new_fcs); Qs_new = np.array(new_Qs)
        sort_idx = np.argsort(fcs_new)
        fcs_new = fcs_new[sort_idx]; Qs_new = Qs_new[sort_idx]
        n_cur   = len(fcs_new)

        gains_new = np.zeros(n_cur)
        for k, fc_new in enumerate(fcs_new):
            best = np.argmin(np.abs(fcs - fc_new))
            if abs(np.log2(fc_new / fcs[best])) < 0.25:
                gains_new[k] = gains[best]

        M_smooth_new = make_smooth_matrix_freq(fcs_new, lam_smooth)
        gains_new, _ = optimize_gains_biquad(
            fcs_new, Qs_new, f_fit, c_fit, w, lam_base, M_smooth_new,
            iters // 2,   # half iters — already warm started
            lam_energy=lam_energy, lam_q=lam_q, lam_gd=lam_gd,
            lam_driver=lam_driver, lam_ph=0.0005, use_minphase=use_minphase,
            raw_avg_db=raw_avg_db, fs=fs)

        rmse_new, _, _, _ = compute_stats_biquad(fcs_new, Qs_new, gains_new, c_fit, f_fit, fs)
        print(f"  Spawning: n={n_cur} (+{added}) RMSE={rmse_new:.4f} dB"
              f" (Δ={rmse_prev - rmse_new:.4f} dB)")

        if rmse_prev - rmse_new < rmse_plateau and n_cur > n_start + n_step:
            print(f"  Spawning: plateau reached (Δ < {rmse_plateau} dB), stopping.")
            break

        fcs = fcs_new; Qs = Qs_new; gains = gains_new
        M_smooth = M_smooth_new; rmse_prev = rmse_new

    return fcs, Qs, gains, M_smooth

def compute_stats_biquad(fcs, Qs, gains, c_fit, f_fit, fs=48000.0):
    _, summed, _, _, _ = biquad_response_and_grad(f_fit, fcs, gains, Qs, fs)
    residual  = summed - c_fit
    rmse      = float(np.sqrt(np.mean(residual ** 2)))
    max_err   = float(np.max(np.abs(residual)))
    max_err_f = float(f_fit[np.argmax(np.abs(residual))])
    return rmse, max_err, max_err_f, summed


def lr_uncertainty_weights(l_pts, r_pts_interp, f_fit, sigma_floor=0.3):
    lr_diff  = np.abs(l_pts - r_pts_interp)
    sigma    = np.maximum(lr_diff, sigma_floor)
    w        = 1.0 / (sigma ** 2)
    w        = w / (w.mean() + 1e-12)
    w        = np.clip(w, 0.1, 10.0)
    return w / w.mean()


def erb_hz(f):
    return 24.7 * (4.37 * np.asarray(f, dtype=float) / 1000.0 + 1.0)


def erb_band_error(f_fit, residual, n_bands=30):
    log_f   = np.log2(np.maximum(f_fit, 1.0))
    edges   = np.linspace(log_f[0], log_f[-1], n_bands + 1)
    errs    = []
    for i in range(n_bands):
        mask = (log_f >= edges[i]) & (log_f < edges[i+1])
        if mask.sum() == 0:
            continue
        errs.append(float(np.sqrt(np.mean(residual[mask] ** 2))))
    return float(np.mean(errs)), float(np.max(errs)) if errs else 0.0



def accuracy_table(f_fit, pred, c_fit):
    checkpoints = [20, 31.5, 50, 63, 80, 100, 125, 160, 200, 250, 315,
                   400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150,
                   4000, 5000, 6300, 8000, 10000, 12500, 16000, 20000]
    fn_c = interpolate.interp1d(f_fit, c_fit, bounds_error=False, fill_value=np.nan)
    fn_p = interpolate.interp1d(f_fit, pred,  bounds_error=False, fill_value=np.nan)
    lines = [f"{'Freq':>7}   {'Target':>7}   {'Fit':>7}   {'Error':>7}   Status",
             "-" * 52]
    for f in checkpoints:
        if f < f_fit[0] or f > f_fit[-1]: continue
        c = float(fn_c(f)); p = float(fn_p(f)); e = p - c
        flag = "✓" if abs(e) < 0.5 else ("⚠" if abs(e) < 1.5 else "✗")
        lines.append(f"{f:7.0f}   {c:+7.2f}   {p:+7.2f}   {e:+7.2f}   {flag}")
    return "\n".join(lines)


def wide_filter_params(f_lo, f_hi):
    fc = np.sqrt(f_lo * f_hi); bw_oct = np.log2(f_hi / f_lo)
    Q  = 1.0 / (2 * np.sinh(np.log(2) / 2 * bw_oct))
    return fc, Q


def apply_boost_cuts(boost_cut_list, fcs, Qs, gains):
    fcs = list(fcs); Qs = list(Qs); gains = list(gains); comments = []
    for (f_lo, f_hi, gain_db, label) in boost_cut_list:
        fc, Q   = wide_filter_params(f_lo, f_hi)
        gain_db = float(np.clip(gain_db, -12, 12))
        fcs.append(fc); Qs.append(Q); gains.append(gain_db)
        comments.append(f"# {label}: {f_lo:.0f}-{f_hi:.0f} Hz | "
                        f"Fc={fc:.1f} Hz  Gain={gain_db:+.2f} dB  Q={Q:.3f}")
        print(f"  {label}: Fc={fc:.1f} Hz  Q={Q:.3f}  Gain={gain_db:+.2f} dB")
    return np.array(fcs), np.array(Qs), np.array(gains), comments


def format_peq(fcs, Qs, gains, preamp, n_opt, extra_comments,
               f_start=20.0, f_end=20000.0, gain_floor=0.1):
    POWERAMP_Q_MAX = 12.0
    GAIN_MIN       = gain_floor
    lines = [
        "# ── Poweramp Parametric EQ ──────────────────────────────",
        "# 1. Set Preamp value in Poweramp equalizer",
        "# 2. Enter each Filter manually in Poweramp's PEQ bands",
        "# 3. Use Peak (PK) filter type for all bands",
        "# ─────────────────────────────────────────────────────────",
        "",
        f"Preamp: {preamp:+.1f} dB",
        ""
    ]
    for i, (fc, Q, g) in enumerate(zip(fcs, Qs, gains), 1):
        fc_out = int(round(fc))
        Q_out  = min(Q, POWERAMP_Q_MAX)
        g_out  = 0.0 if abs(g) < GAIN_MIN else g
        tag    = "  # boost/cut" if i > n_opt else ""
        lines.append(f"Filter {i:2d}: ON PK Fc {fc_out:6d} Hz  "
                     f"Gain {g_out:+.2f} dB  Q {Q_out:.2f}{tag}")
    if extra_comments:
        lines += ["", "# Boost/cut filter details:"] + extra_comments
    return "\n".join(lines)


def format_geq(fcs, corr_fn):
    gains = np.clip(corr_fn(fcs), -12, 12)
    max_b = max(0.0, float(np.max(gains)))
    parts = "; ".join(f"{fc:.2f} {g:.2f}" for fc, g in zip(fcs, gains))
    return f"Preamp: {-(max_b+0.5):+.1f} dB\nGraphicEQ: {parts}"


def compute_true_peak(fcs, Qs, gains, fs=48000.0, n_points=32000):
    f_grid = np.logspace(np.log10(20.0), np.log10(min(fs / 2.0, 20000.0)), n_points)
    summed = np.zeros(n_points)
    for fc, Q, g in zip(fcs, Qs, gains):
        summed += biquad_peak_db(f_grid, fc, g, Q, fs=fs)
    return float(np.max(summed))


def simulate_post_eq(l_pts, r_pts, fcs, Qs, gains, preamp, out_path, fs=48000.0):
    r_fn    = make_interp(r_pts)
    avg_db  = (l_pts[:, 1] + r_fn(l_pts[:, 0])) / 2
    freqs   = l_pts[:, 0]
    eq_resp = np.zeros(len(freqs))
    for fc, Q, g in zip(fcs, Qs, gains):
        eq_resp += biquad_peak_db(freqs, fc, g, Q, fs=fs)
    eq_resp += preamp
    lines = [f"# Post-EQ simulated (fs={fs:.0f} Hz, exact biquad)", "# freq,dB"]
    for f, d in zip(freqs, avg_db + eq_resp):
        lines.append(f"{f},{d:.6f}")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Post-EQ simulation -> {out_path}")


def compute_ir_diagnostic(fcs, Qs, gains, fs=48000.0, n_ir=4096):
    from scipy.signal import lfilter
    x = np.zeros(n_ir); x[0] = 1.0

    y = x.copy()
    for fc, Q, g in zip(fcs, Qs, gains):
        A     = 10 ** (g / 40.0)
        w0    = 2 * np.pi * fc / fs
        sin0  = np.sin(w0); cos0 = np.cos(w0)
        alpha = sin0 / (2.0 * Q)
        b0 = (1 + alpha * A) / (1 + alpha / A)
        b1 = (-2 * cos0)     / (1 + alpha / A)
        b2 = (1 - alpha * A) / (1 + alpha / A)
        a1 = (-2 * cos0)     / (1 + alpha / A)
        a2 = (1 - alpha / A) / (1 + alpha / A)
        y  = lfilter([b0, b1, b2], [1.0, a1, a2], y)

    energy = y ** 2
    total_energy = energy.sum() + 1e-30
    t_ms = np.arange(n_ir) / fs * 1000.0  # time axis in ms

    temporal_centroid_ms = float(np.sum(t_ms * energy) / total_energy)

    early_samples = int(0.002 * fs)
    early_energy  = energy[:early_samples].sum()
    early_energy_ratio = float(early_energy / total_energy)

    N_sf    = 2048
    ir_fft  = np.fft.rfft(y, n=N_sf)
    freqs_sf = np.fft.rfftfreq(N_sf, d=1.0/fs)
    mask_sf  = (freqs_sf >= 20.0) & (freqs_sf <= 20000.0)
    psd      = np.abs(ir_fft[mask_sf]) ** 2 + 1e-30
    log_mean = np.mean(np.log(psd))
    arith_mean = np.mean(psd)
    spectral_flatness = float(np.exp(log_mean) / (arith_mean + 1e-30))
    spectral_flatness_db = float(10 * np.log10(max(spectral_flatness, 1e-12)))

    win_samples = int(0.004 * fs)
    hop_samples = win_samples // 2
    n_frames    = max(1, (len(y) - win_samples) // hop_samples)
    win_h       = np.hanning(win_samples)
    frame_flat  = []
    for fi in range(n_frames):
        start     = fi * hop_samples
        frame     = y[start:start + win_samples] * win_h
        ffr       = np.fft.rfft(frame, n=512)
        ffr_f     = np.fft.rfftfreq(512, d=1.0/fs)
        fm        = (ffr_f >= 200.0) & (ffr_f <= 16000.0)
        if fm.sum() < 4:
            continue
        fpsd      = np.abs(ffr[fm]) ** 2 + 1e-30
        frame_flat.append(float(np.exp(np.mean(np.log(fpsd))) / (np.mean(fpsd) + 1e-30)))
    min_flatness_db = float(10 * np.log10(
        max(min(frame_flat) if frame_flat else spectral_flatness, 1e-12)))

    schroeder = np.cumsum(energy[::-1])[::-1]  # backward cumulative sum
    schroeder_db = 10 * np.log10(np.maximum(schroeder / (schroeder[0] + 1e-30), 1e-12))
    decay_idx = np.searchsorted(-schroeder_db, 20.0)  # first point ≤ -20 dB
    decay_time_20db_ms = float(t_ms[min(decay_idx, n_ir - 1)])

    ir_peak_db = float(20 * np.log10(max(np.max(np.abs(y)), 1e-10)))

    peak_energy_ratio = float(np.max(energy) / (total_energy + 1e-30))
    peak_energy_ratio_db = float(10 * np.log10(max(peak_energy_ratio, 1e-12)))

    return {
        "temporal_centroid_ms"  : temporal_centroid_ms,
        "early_energy_ratio"    : early_energy_ratio,
        "decay_time_20db_ms"    : decay_time_20db_ms,
        "ir_peak_db"            : ir_peak_db,
        "peak_energy_ratio_db"  : peak_energy_ratio_db,
        "spectral_flatness_db"  : spectral_flatness_db,
        "min_flatness_db"       : min_flatness_db,
    }


def print_ir_diagnostic(fcs, Qs, gains, fs=48000.0):
    d = compute_ir_diagnostic(fcs, Qs, gains, fs=fs)
    eer_pct = d["early_energy_ratio"] * 100.0

    if   eer_pct > 85: eer_label = "excellent (laser tight)"
    elif eer_pct > 70: eer_label = "good (clean transient)"
    elif eer_pct > 55: eer_label = "moderate (slight smear)"
    else:              eer_label = "poor (diffuse / ringy)"

    dt = d["decay_time_20db_ms"]
    if   dt <  5: dt_label = "very fast"
    elif dt < 15: dt_label = "fast"
    elif dt < 30: dt_label = "moderate"
    else:         dt_label = "slow (fatigue risk)"

    per_db = d["peak_energy_ratio_db"]
    if   per_db > -13: per_label = "excellent (laser attack)"
    elif per_db > -20: per_label = "good (sharp transient)"
    elif per_db > -28: per_label = "moderate (some smear)"
    else:              per_label = "diffuse (blurred attack)"

    sf_db = d["spectral_flatness_db"]
    if   sf_db > -6:  sf_label = "broadband (resolving, low fatigue)"
    elif sf_db > -12: sf_label = "moderate (balanced)"
    elif sf_db > -20: sf_label = "tonal (some resonance)"
    else:             sf_label = "narrow (metallic / fatiguing risk)"

    mf_db = d["min_flatness_db"]
    if   mf_db > -8:  mf_label = "clean (no localized ringing)"
    elif mf_db > -15: mf_label = "mild tonal frame"
    elif mf_db > -22: mf_label = "tonal region detected"
    else:             mf_label = "ringing region — fatigue risk"

    print("\nIR Diagnostic [v14] (time-domain cascade quality):")
    print(f"  Temporal centroid    : {d['temporal_centroid_ms']:.2f} ms")
    print(f"  Early energy (0-2ms) : {eer_pct:.1f}%  → {eer_label}")
    print(f"  Decay to -20 dB      : {dt:.1f} ms  → {dt_label}")
    print(f"  Peak/energy ratio    : {per_db:.1f} dB  → {per_label}")
    print(f"  Spectral flatness    : {sf_db:.1f} dB  → {sf_label}")
    print(f"  Min frame flatness   : {mf_db:.1f} dB  → {mf_label}")
    print(f"  IR peak              : {d['ir_peak_db']:.2f} dBFS")


class BoostCutAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        try:
            f_lo, f_hi, gain = float(values[0]), float(values[1]), float(values[2])
        except (ValueError, IndexError):
            parser.error(f"{option_string} requires: F_LO F_HI GAIN_DB")
        if option_string == "--cut": gain = -abs(gain)
        label = ("boost" if gain >= 0 else "cut") + f" {f_lo:.0f}-{f_hi:.0f}Hz"
        lst = getattr(namespace, self.dest) or []
        lst.append((f_lo, f_hi, gain, label))
        setattr(namespace, self.dest, lst)


def smart_norm_freq(l_pts, r_fn, lo=500.0, hi=800.0):
    avg = (l_pts[:, 1] + r_fn(l_pts[:, 0])) / 2
    mask = (l_pts[:, 0] >= lo) & (l_pts[:, 0] <= hi)
    if mask.sum() < 3:
        return np.sqrt(lo * hi)

    freqs_w = l_pts[mask, 0]
    avg_w   = avg[mask]
    log_f   = np.log2(l_pts[:, 0])
    log_f_w = np.log2(freqs_w)
    dlog    = np.mean(np.diff(np.log2(l_pts[:, 0])))
    win_pts = max(int(0.1 / dlog), 2)  # ±0.1 oct window

    best_var = np.inf
    best_f   = float(np.sqrt(lo * hi))

    for i, (f, _) in enumerate(zip(freqs_w, avg_w)):
        log_f_i = np.log2(f)
        win_mask = np.abs(log_f - log_f_i) <= 0.1
        if win_mask.sum() < 3:
            continue
        var = float(np.var(avg[win_mask]))
        if var < best_var:
            best_var = var
            best_f   = f

    return float(best_f)


def auto_freq_end(l_pts, r_fn, floor_db=-25.0, ref_lo=500.0, ref_hi=2000.0,
                  hard_max=20000.0):
    avg   = (l_pts[:, 1] + r_fn(l_pts[:, 0])) / 2
    freqs = l_pts[:, 0]

    ref_mask = (freqs >= ref_lo) & (freqs <= ref_hi)
    if ref_mask.sum() == 0:
        return hard_max
    ref_level = float(np.median(avg[ref_mask]))

    hf_mask = freqs >= 5000.0
    if hf_mask.sum() == 0:
        return hard_max

    hf_f   = freqs[hf_mask]
    hf_avg = avg[hf_mask]

    detected_floor = hard_max
    for f, a in zip(hf_f[::-1], hf_avg[::-1]):
        if a >= ref_level + floor_db:
            detected_floor = float(f)
            break

    hf_drop_db = -15.0  # dB drop from local HF peak → consider rolled off
    peak_mask  = (freqs >= 5000.0) & (freqs <= 12000.0)
    if peak_mask.sum() > 0:
        hf_peak = float(np.max(avg[peak_mask]))
        scan_mask = freqs >= 10000.0
        scan_f    = freqs[scan_mask]
        scan_avg  = avg[scan_mask]

        detected_hfpeak = hard_max
        for f, a in zip(scan_f[::-1], scan_avg[::-1]):
            if a >= hf_peak + hf_drop_db:
                detected_hfpeak = float(f)
                break
    else:
        detected_hfpeak = hard_max

    detected = min(detected_floor, detected_hfpeak)
    return float(np.clip(detected, 8000.0, hard_max))


def auto_detect_files(target_path=None):
    import glob, re

    candidates = [f for f in glob.glob("*.txt")
                  if f != target_path and not f.startswith("autoeq")]

    if len(candidates) == 0:
        raise FileNotFoundError(
            "No .txt measurement files found in current directory.\n"
            "Place L and R measurement files here or pass them as arguments:\n"
            "  python autoeq17-1.py LEFT.txt RIGHT.txt"
        )

    l_files, r_files = [], []
    for f in candidates:
        name = f.lower()
        if re.search(r'[_\s\(\[]l[_\s\)\]\.]|\bl\b|left|_l\.', name):
            l_files.append(f)
        if re.search(r'[_\s\(\[]r[_\s\)\]\.]|\br\b|right|_r\.', name):
            r_files.append(f)

    if l_files and r_files:
        return l_files[0], r_files[0]

    if len(candidates) == 2:
        candidates.sort()
        return candidates[0], candidates[1]

    raise FileNotFoundError(
        f"Found {len(candidates)} .txt file(s) but could not determine L/R pair.\n"
        f"Files found: {candidates}\n"
        "Rename files to include '_L'/'_R' or pass them explicitly:\n"
        "  python autoeq17-1.py LEFT.txt RIGHT.txt"
    )


def load_config(path="autoeq_config.txt"):
    cfg = {}
    if not os.path.exists(path):
        return cfg
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                k, _, v = line.partition("=")
                cfg[k.strip().lower()] = v.strip()
    return cfg


def main():
    parser = argparse.ArgumentParser(
        description="AutoEQ v17 — IEM correction pipeline → Poweramp 64-band PEQ. See README.md for usage.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Example:\n  python3 autoeq17-1.py left.txt right.txt -o result.txt\n"
               "  python3 autoeq17-1.py left.txt right.txt --target my_target.txt -o result.txt"
    )
    parser.add_argument("left",          nargs="?", default=None,
                        help="Left channel measurement file (auto-detected if omitted)")
    parser.add_argument("right",         nargs="?", default=None,
                        help="Right channel measurement file (auto-detected if omitted)")
    parser.add_argument("--target", "-t", default=None,
                        help="Target curve file (default: built-in personal target)")
    parser.add_argument("--freq-start",  type=float, default=20,
                        help="Optimization lower bound Hz (default: 20). "
                             "Minimum 20 Hz — Poweramp's lowest supported frequency. "
                             "Measurements below 20 Hz are still used for normalization.")
    parser.add_argument("--freq-end",    type=float, default=None,
                        help="Optimization upper bound Hz (default: auto-detect from rolloff)")
    parser.add_argument("--norm-freq",   type=float, default=None,
                        help="Normalization anchor Hz (default: auto-select flattest point 500-800 Hz)")
    parser.add_argument("--norm-lo",     type=float, default=300)
    parser.add_argument("--norm-hi",     type=float, default=700)
    parser.add_argument("--norm-mode",   default="perceptual",
                        choices=["freq", "energy", "perceptual"],
                        help="Normalization mode: 'perceptual' = re-center by perceptual mean (default), "
                             "'freq' = single point, 'energy' = loudness-neutral preamp")
    parser.add_argument("--bands",       type=int,   default=64)
    parser.add_argument("--lambda",      type=float, default=0.15, dest="lam")
    parser.add_argument("--lam-smooth",  type=float, default=0.10)
    parser.add_argument("--lam-energy",  type=float, default=0.05)
    parser.add_argument("--iters",       type=int,   default=1500,
                        help="Biquad IRLS iterations (default: 1500)")
    parser.add_argument("--joint-iters", type=int,   default=800)
    parser.add_argument("--fc-range",    type=float, default=0.15)
    parser.add_argument("--no-realloc",  action="store_true")
    parser.add_argument("--spawn",       action="store_true",
                        help="[v15] Iterative band spawning: start 32 bands, grow to "
                             "--bands by spawning at residual hotspots. Slower but "
                             "potentially better band distribution.")
    parser.add_argument("--lam-q",       type=float, default=0.02,
                        help="Q log-penalty weight (default: 0.02)")
    parser.add_argument("--lam-gd",      type=float, default=0.001,
                        help="Group delay curvature penalty (default: 0.001)")
    parser.add_argument("--lam-driver",  type=float, default=0.5,
                        help="HF driver capability constraint (default: 0.5)")
    parser.add_argument("--no-minphase", action="store_true",
                        help="Skip minimum phase reconstruction")
    parser.add_argument("--lam-ph",      type=float, default=0.0,
                        help="Phase slope regularizer weight (default: 0 = disabled for speed). "
                             "Enable with e.g. 0.0005 for phase-aware tuning.")
    parser.add_argument("--fs",           type=float, default=48000.0,
                        help="Target device sample rate Hz (default: 48000). "
                             "Use 96000 for high-res DAC/DAP.")
    parser.add_argument("--gain-floor",     type=float, default=0.10,
                        help="Minimum absolute gain to write to output (default: 0.10 dB). "
                             "Bands below this threshold are written as 0.00.")
    parser.add_argument("--presence-boost", type=float, default=2.2,
                        help="Band density multiplier in 2–8 kHz presence region (default: 2.2). "
                             "Higher = more bands allocated to presence/vocal zone.")
    parser.add_argument("--treble-boost",   type=float, default=1.8,
                        help="Band density multiplier in 8–20 kHz treble region (default: 1.8). "
                             "Higher = more bands allocated to treble/air zone.")
    parser.add_argument("--no-warp",     action="store_true")
    parser.add_argument("--geq",          action="store_true",
                        help="Also output GraphicEQ section (disabled by default, Poweramp uses PEQ only)")
    parser.add_argument("--no-simulate", action="store_true",
                        help="Skip post-EQ simulation output (default: always generated)")
    parser.add_argument("--output", "-o", default="autoeq_result.txt")
    parser.add_argument("--boost", nargs=3, metavar=("F_LO","F_HI","DB"),
                        action=BoostCutAction, dest="boost_cuts", default=[])
    parser.add_argument("--cut",   nargs=3, metavar=("F_LO","F_HI","DB"),
                        action=BoostCutAction, dest="boost_cuts")
    args = parser.parse_args()

    cfg = load_config()
    if cfg:
        print(f"Config     : loaded autoeq_config.txt")
        if args.left  is None and "left"   in cfg: args.left   = cfg["left"]
        if args.right is None and "right"  in cfg: args.right  = cfg["right"]
        if args.target is None and "target" in cfg: args.target = cfg["target"]
        if args.output == "autoeq_result.txt" and "output" in cfg:
            args.output = cfg["output"]
        if args.norm_freq is None and "norm_freq" in cfg:
            try: args.norm_freq = float(cfg["norm_freq"])
            except ValueError: pass
        if "norm_mode" in cfg: args.norm_mode = cfg["norm_mode"]
        if args.freq_end is None and "freq_end" in cfg:
            try: args.freq_end = float(cfg["freq_end"])
            except ValueError: pass

    if args.left is None or args.right is None:
        try:
            detected_l, detected_r = auto_detect_files(args.target)
            if args.left  is None: args.left  = detected_l
            if args.right is None: args.right = detected_r
            print(f"Auto-detect: L={args.left} | R={args.right}")
        except FileNotFoundError as e:
            print(f"ERROR: {e}"); sys.exit(1)

    if args.output == "autoeq_result.txt":
        import re as _re
        def _clean(s):
            s = Path(s).stem
            s = _re.sub(r'\s*[\[\(]?[LR][\]\)]?\s*$', '', s, flags=_re.IGNORECASE).strip()
            s = _re.sub(r'[_\-]\s*[LR]\s*$', '', s, flags=_re.IGNORECASE).strip()
            return s.upper()
        iem_name  = _clean(args.left)
        if args.target:
            tgt_name = Path(args.target).stem.upper()
        else:
            tgt_name = "EUVONY REF"
        safe = _re.sub(r'[^\w\s\-]', '', f"{iem_name} - {tgt_name}").strip()
        args.output = safe + ".txt"
        print(f"Output     : {args.output} (auto-named)")

    print(f"Loading L : {args.left}")
    l_pts = parse_txt(args.left)
    print(f"  -> {len(l_pts)} pts | {l_pts[0,0]:.1f}-{l_pts[-1,0]:.1f} Hz")
    print(f"Loading R : {args.right}")
    r_pts = parse_txt(args.right)
    print(f"  -> {len(r_pts)} pts | {r_pts[0,0]:.1f}-{r_pts[-1,0]:.1f} Hz")

    r_fn_early = make_interp(r_pts)

    if args.norm_freq is None:
        args.norm_freq = smart_norm_freq(l_pts, r_fn_early, lo=500.0, hi=800.0)
        print(f"Norm freq  : auto-selected {args.norm_freq:.1f} Hz (flattest region 500-800 Hz)")
    else:
        print(f"Norm freq  : {args.norm_freq:.1f} Hz (manual)")

    if args.freq_end is None:
        args.freq_end = auto_freq_end(l_pts, r_fn_early)
        print(f"Freq end   : auto-detected {args.freq_end:.0f} Hz from rolloff")
    else:
        print(f"Freq end   : {args.freq_end:.0f} Hz (manual)")

    if args.freq_start >= args.freq_end:
        print("ERROR: --freq-start must be < --freq-end"); sys.exit(1)

    if args.target is None:
        tgt_pts = builtin_target_pts()
        tgt_label = "built-in Euvony Reference"
        print("Loading T : [built-in Euvony Reference target]")
        print("            Euvony Reference v1r6 | vocal-centric detail-hunter | see README")
    else:
        print(f"Loading T : {args.target}")
        tgt_pts = parse_txt(args.target)
        tgt_label = os.path.basename(args.target)
    print(f"  -> {len(tgt_pts)} pts | {tgt_pts[0,0]:.1f}-{tgt_pts[-1,0]:.1f} Hz")

    r_fn    = r_fn_early; tgt_fn = make_interp(tgt_pts)
    avg_db  = (l_pts[:, 1] + r_fn(l_pts[:, 0])) / 2
    raw_corr = tgt_fn(l_pts[:, 0]) - avg_db

    if args.norm_mode == "perceptual":
        w_perc_norm = perceptual_weights(l_pts[:, 0])
        w_perc_norm = w_perc_norm / w_perc_norm.sum()
        norm_offset = float(np.sum(raw_corr * w_perc_norm))
        corr        = raw_corr - norm_offset
        print(f"Perceptual norm    : weighted mean offset = {norm_offset:+.3f} dB → centered to 0")
    else:
        norm_offset = compute_norm_offset(l_pts, r_fn, tgt_fn,
                                          args.norm_freq, args.norm_lo, args.norm_hi)
        corr = raw_corr - norm_offset

        if args.norm_mode == "energy":
            print(f"Energy norm        : anchor @ {args.norm_freq:.0f} Hz | delta={norm_offset:+.3f} dB")
        else:
            print(f"Freq norm          : anchor @ {args.norm_freq:.0f} Hz | delta={norm_offset:+.3f} dB")

    corr_fn = make_interp(np.column_stack([l_pts[:, 0], corr]))
    print(f"Correction range   : {corr.min():.2f} to {corr.max():.2f} dB")

    POWERAMP_FC_MIN = 20.0
    f_start = max(args.freq_start, POWERAMP_FC_MIN, float(l_pts[0, 0]))
    f_end   = min(args.freq_end,   float(l_pts[-1, 0]))
    mask    = (l_pts[:, 0] >= f_start) & (l_pts[:, 0] <= f_end)
    f_fit   = l_pts[mask, 0]; c_fit = corr[mask]
    raw_avg_fit = avg_db[mask]
    print(f"Fit range  : {f_start:.1f}-{f_end:.1f} Hz ({mask.sum()} pts)")

    r_pts_interp_fit = r_fn(f_fit)
    l_pts_fit        = l_pts[mask, 1]
    w_uncertainty    = lr_uncertainty_weights(l_pts_fit, r_pts_interp_fit, f_fit)
    weights_perc     = perceptual_weights(f_fit)
    weights          = weights_perc * w_uncertainty
    weights          = weights / (weights.mean() + 1e-12)
    n_stable = int(np.sum(w_uncertainty >= 0.5))
    print(f"L/R uncertainty weighting: {n_stable}/{len(f_fit)} pts stable (L-R < 0.6 dB)")

    n_bc    = len(args.boost_cuts) if args.boost_cuts else 0
    if args.bands > 64:
        print(f"WARNING: --bands {args.bands} exceeds Poweramp 64-band limit, clamping to 64")
        args.bands = 64
    n_bands = min(args.bands, 64 - n_bc)
    use_warp   = not args.no_warp
    warp_label = "warped (presence boost 2k-8k)" if use_warp else "plain logspace"

    if args.spawn:
        n_start_spawn = max(16, n_bands // 2)
        print(f"\nBands      : spawn mode {n_start_spawn}→{n_bands} + {n_bc} boost/cut")
        print(f"v15 params : lam={args.lam} lam_s={args.lam_smooth} "
              f"lam_e={args.lam_energy} fc_range={args.fc_range}")
        print(f"Running iterative band spawning [v15]...")
        fcs, Qs, gains, M_smooth = iterative_band_spawn(
            f_fit, c_fit, weights, args.lam, args.lam_smooth,
            args.lam_energy, args.lam_q, args.lam_gd, args.lam_driver,
            f_start, f_end, not args.no_minphase, raw_avg_fit,
            args.iters, max_iters_spawn=6,
            n_start=n_start_spawn, n_max=n_bands, n_step=max(4, n_bands//8),
            fs=args.fs)
        N = len(fcs)
        converged = True
        rmse_irls, max_err_irls, max_err_f_irls, pred_irls = compute_stats_biquad(
            fcs, Qs, gains, c_fit, f_fit)
        print(f"Spawn final: {N} bands | RMSE={rmse_irls:.4f} dB"
              f" | Max={max_err_irls:.4f} dB @ {max_err_f_irls:.0f} Hz")
    else:
        fcs, Qs = make_bands(f_start, f_end, n_bands=n_bands, warp=use_warp,
                             presence_boost=args.presence_boost,
                             treble_boost=args.treble_boost)
        N = len(fcs)
        print(f"\nBands      : {N} optimizer + {n_bc} boost/cut = {N+n_bc} total")
        print(f"Dist       : {warp_label} | Q: {Qs.min():.2f}-{Qs.max():.2f} (adaptive)")
        print(f"v6 params  : lam={args.lam} lam_s={args.lam_smooth} "
              f"lam_e={args.lam_energy} fc_range={args.fc_range}")
        M_smooth = make_smooth_matrix_freq(fcs, args.lam_smooth)

        print(f"Running exact-biquad IRLS (iters={args.iters})...")
        gains, converged = optimize_gains_biquad(
            fcs, Qs, f_fit, c_fit, weights,
            args.lam, M_smooth, args.iters, lam_energy=args.lam_energy,
            lam_q=args.lam_q, lam_gd=args.lam_gd, lam_driver=args.lam_driver, lam_ph=args.lam_ph,
            use_minphase=not args.no_minphase, raw_avg_db=raw_avg_fit, fs=args.fs
        )
        rmse_irls, max_err_irls, max_err_f_irls, pred_irls = compute_stats_biquad(
            fcs, Qs, gains, c_fit, f_fit)
        print(f"Converged  : {converged}")
        print(f"RMSE (biquad): {rmse_irls:.4f} dB | Max: {max_err_irls:.4f} dB @ {max_err_f_irls:.0f} Hz")

        if use_warp:
            fcs2, Qs2 = make_bands(f_start, f_end, n_bands=n_bands, warp=False)
            M2 = make_smooth_matrix_freq(fcs2, args.lam_smooth)
            gains2, _ = optimize_gains_biquad(
                fcs2, Qs2, f_fit, c_fit, weights,
                args.lam, M2, args.iters // 2, lam_energy=args.lam_energy,
                lam_q=args.lam_q, lam_gd=args.lam_gd, lam_driver=args.lam_driver, lam_ph=args.lam_ph,
                use_minphase=not args.no_minphase, raw_avg_db=raw_avg_fit, fs=args.fs
            )
            rmse2 = compute_stats_biquad(fcs2, Qs2, gains2, c_fit, f_fit)[0]
            if rmse2 < rmse_irls:
                fcs, Qs, gains, M_smooth = fcs2, Qs2, gains2, M2
                rmse_irls, max_err_irls, max_err_f_irls, pred_irls = compute_stats_biquad(
                    fcs, Qs, gains, c_fit, f_fit)
                print(f"Multi-start: flat seed better ({rmse2:.4f} < {rmse_irls:.4f}), using flat")
            else:
                print(f"Multi-start: warped seed kept ({rmse_irls:.4f} <= {rmse2:.4f})")

    if not args.no_realloc:
        print("Iterative band reallocation [v16: greedy loop until plateau]...")
        rmse_realloc = compute_stats_biquad(fcs, Qs, gains, c_fit, f_fit)[0]
        for _realloc_iter in _progress(range(4), desc="Realloc", leave=False):  # max 4 passes
            fcs_r, Qs_r, n_moved = adaptive_band_realloc(
                f_fit, c_fit, fcs, Qs, gains, waste_thresh=0.5, residual_min=0.5)
            if n_moved == 0:
                print(f"  Pass {_realloc_iter+1}: no bands moved, stopping.")
                break
            print(f"  Pass {_realloc_iter+1}: {n_moved} band(s) reallocated -> re-solving...")
            M_smooth_r = make_smooth_matrix_freq(fcs_r, args.lam_smooth)
            gains_r, _ = optimize_gains_biquad(
                fcs_r, Qs_r, f_fit, c_fit, weights,
                args.lam, M_smooth_r, args.iters, lam_energy=args.lam_energy,
                lam_q=args.lam_q, lam_gd=args.lam_gd, lam_driver=args.lam_driver, lam_ph=args.lam_ph,
                use_minphase=not args.no_minphase, raw_avg_db=raw_avg_fit)
            rmse_r, max_err_r, max_err_f_r, _ = compute_stats_biquad(
                fcs_r, Qs_r, gains_r, c_fit, f_fit)
            print(f"  RMSE: {rmse_r:.4f} dB | Max: {max_err_r:.4f} dB @ {max_err_f_r:.0f} Hz")
            if rmse_r < rmse_realloc:
                fcs, Qs, gains, M_smooth = fcs_r, Qs_r, gains_r, M_smooth_r
                rmse_realloc = rmse_r
            else:
                print(f"  No improvement ({rmse_realloc:.4f} <= {rmse_r:.4f}), stopping.")
                break

    if args.joint_iters > 0:
        rmse_pre = compute_stats_biquad(fcs, Qs, gains, c_fit, f_fit)[0]
        print(f"Joint fc+Q+gain [v6: ~0 warm start gap] (iters={args.joint_iters})...")
        gains_j, fcs_j, Qs_j, j_ok = optimize_joint(
            gains, fcs, Qs, f_fit, c_fit, weights,
            args.lam, M_smooth, args.joint_iters,
            fc_range=args.fc_range, lam_energy=args.lam_energy,
        )
        gains_j = np.clip(gains_j, -12, 12)
        sort_idx = np.argsort(fcs_j)
        fcs_j = fcs_j[sort_idx]; Qs_j = Qs_j[sort_idx]; gains_j = gains_j[sort_idx]
        rmse_j, max_err_j, max_err_f_j, pred_j = compute_stats_biquad(
            fcs_j, Qs_j, gains_j, c_fit, f_fit)
        print(f"Converged  : {j_ok}")
        print(f"RMSE joint : {rmse_j:.4f} dB | Max: {max_err_j:.4f} dB @ {max_err_f_j:.0f} Hz")

        if rmse_j < rmse_pre:
            gains, fcs, Qs = gains_j, fcs_j, Qs_j
            M_smooth = make_smooth_matrix_freq(fcs, args.lam_smooth)
            pred = pred_j; rmse = rmse_j; max_err = max_err_j; max_err_f = max_err_f_j
            print(f"  Joint accepted  ({rmse_pre:.4f} -> {rmse_j:.4f} dB ✓)")

            if not args.no_realloc:
                print("Post-joint band reallocation [v9: tighter threshold 0.3]...")
                fcs_pj, Qs_pj, n_moved_pj = adaptive_band_realloc(
                    f_fit, c_fit, fcs, Qs, gains,
                    waste_thresh=0.3, residual_min=0.4)
                if n_moved_pj > 0:
                    print(f"  {n_moved_pj} band(s) reallocated post-joint -> re-solving...")
                    M_smooth_pj = make_smooth_matrix_freq(fcs_pj, args.lam_smooth)
                    gains_pj, _ = optimize_gains_biquad(
                        fcs_pj, Qs_pj, f_fit, c_fit, weights,
                        args.lam, M_smooth_pj, args.iters, lam_energy=args.lam_energy,
                        lam_q=args.lam_q, lam_gd=args.lam_gd, lam_driver=args.lam_driver, lam_ph=args.lam_ph,
                        use_minphase=not args.no_minphase, raw_avg_db=raw_avg_fit)
                    rmse_pj, max_err_pj, max_err_f_pj, pred_pj = compute_stats_biquad(
                        fcs_pj, Qs_pj, gains_pj, c_fit, f_fit)
                    print(f"  RMSE post-joint-realloc: {rmse_pj:.4f} dB"
                          f" | Max: {max_err_pj:.4f} dB @ {max_err_f_pj:.0f} Hz")
                    if rmse_pj < rmse:
                        fcs, Qs, gains = fcs_pj, Qs_pj, gains_pj
                        M_smooth = M_smooth_pj
                        pred = pred_pj; rmse = rmse_pj
                        max_err = max_err_pj; max_err_f = max_err_f_pj
                        print(f"  Post-joint realloc accepted ({rmse_j:.4f} -> {rmse_pj:.4f} dB ✓)")
                    else:
                        print(f"  Post-joint realloc rejected (joint={rmse_j:.4f} <= pj={rmse_pj:.4f}, joint retained)")
                else:
                    print("  No bands reallocated post-joint.")
        else:
            pred = pred_irls; rmse = rmse_irls; max_err = max_err_irls; max_err_f = max_err_f_irls
            print(f"  Joint rejected  (IRLS={rmse_pre:.4f} <= joint={rmse_j:.4f}, IRLS retained)")
    else:
        pred = pred_irls; rmse = rmse_irls; max_err = max_err_irls; max_err_f = max_err_f_irls

    diffs = np.diff(gains)
    print(f"Smoothness : max adj delta={np.max(np.abs(diffs)):.3f} dB"
          f" | RMS delta={np.sqrt(np.mean(diffs**2)):.3f} dB")

    _, final_summed, _, _, _ = biquad_response_and_grad(f_fit, fcs, gains, Qs)
    final_residual = final_summed - c_fit
    erb_mean, erb_max = erb_band_error(f_fit, final_residual)
    print(f"ERB-band   : mean={erb_mean:.4f} dB | worst-band={erb_max:.4f} dB")

    _fc_min  = 20.0
    _fc_step = 1.0
    _gain_min = 0.1   # minimum audible/useful gain

    sort_idx = np.argsort(fcs)
    fcs   = fcs[sort_idx].copy()
    Qs    = Qs[sort_idx].copy()
    gains = gains[sort_idx].copy()

    n_clamped = int(np.sum(fcs < _fc_min))
    fcs = np.maximum(fcs, _fc_min)
    for i in range(1, len(fcs)):
        if fcs[i] < fcs[i-1] + _fc_step:
            fcs[i] = fcs[i-1] + _fc_step
    if n_clamped > 0:
        print(f"FC clamp   : {n_clamped} band(s) clamped/spread from {_fc_min:.0f} Hz "
              f"with {_fc_step:.0f} Hz minimum spacing")

    _, summed_pre, _, _, _ = biquad_response_and_grad(f_fit, fcs, gains, Qs)
    residual_pre = np.abs(summed_pre - c_fit)

    wasted_idx = np.where(np.abs(gains) < _gain_min)[0]
    n_wasted   = len(wasted_idx)

    if n_wasted > 0:
        non_wasted_fcs = fcs[np.abs(gains) >= _gain_min]
        peaks = _local_maxima(residual_pre, min_val=0.2)
        if len(peaks) == 0:
            peaks = np.argsort(residual_pre)[-n_wasted:][::-1]
        else:
            peaks = peaks[np.argsort(-residual_pre[peaks])]

        moved = 0
        for wi in wasted_idx:
            if moved >= len(peaks):
                break
            for pk in peaks:
                f_hot = float(f_fit[pk])
                if non_wasted_fcs.size > 0 and np.min(np.abs(np.log2(non_wasted_fcs / max(f_hot,1)))) < 0.2:
                    continue
                if f_hot < _fc_min:
                    f_hot = _fc_min
                fcs[wi]   = f_hot
                Qs[wi]    = adaptive_q(f_hot)
                gains[wi] = 0.0   # optimizer will refine in mini-solve
                non_wasted_fcs = np.append(non_wasted_fcs, f_hot)
                moved += 1
                break

        if moved > 0:
            sort_idx2 = np.argsort(fcs)
            fcs = fcs[sort_idx2]; Qs = Qs[sort_idx2]; gains = gains[sort_idx2]
            M_smooth_c = make_smooth_matrix_freq(fcs, args.lam_smooth)
            gains, _ = optimize_gains_biquad(
                fcs, Qs, f_fit, c_fit, weights,
                args.lam, M_smooth_c, min(args.iters // 3, 500),
                lam_energy=args.lam_energy, lam_q=args.lam_q,
                lam_gd=args.lam_gd, lam_driver=args.lam_driver,
                lam_ph=args.lam_ph,
                use_minphase=not args.no_minphase,
                raw_avg_db=raw_avg_fit, fs=args.fs)
            rmse_c, max_err_c, max_err_f_c, pred = compute_stats_biquad(
                fcs, Qs, gains, c_fit, f_fit)
            rmse = rmse_c; max_err = max_err_c; max_err_f = max_err_f_c
            print(f"Band cleanup: {n_wasted} wasted band(s) → {moved} relocated "
                  f"| RMSE after: {rmse_c:.4f} dB")
        else:
            print(f"Band cleanup: {n_wasted} band(s) with |gain| < {_gain_min} dB "
                  f"(no hotspot found to relocate)")

    f_grid  = np.logspace(np.log10(max(f_start, 20)),
                          np.log10(min(f_end, 20000)), 2000)
    eq_resp = np.zeros(len(f_grid))
    for fc, Q, g in zip(fcs, Qs, gains):
        eq_resp += biquad_peak_db(f_grid, fc, g, Q)
    log_f = np.log10(f_grid)
    pw    = 1.0 / (np.gradient(log_f) + 1e-12)
    pw   /= pw.sum()
    mean_gain = float(np.sum(eq_resp * pw))

    n_opt = N

    raw_peak = compute_true_peak(fcs, Qs, gains)

    if args.norm_mode == "energy":
        preamp_clip   = -(raw_peak + 0.5)          # clipping guard
        preamp_energy = -(mean_gain + 0.5)          # loudness-neutral
        preamp = round(min(preamp_clip, preamp_energy), 1)
        print(f"Preamp     : true peak={raw_peak:.3f} dB | mean gain={mean_gain:.3f} dB"
              f" | energy-neutral={preamp_energy:.1f} dB"
              f" | clip guard={preamp_clip:.1f} dB -> {preamp:+.1f} dB")
    else:
        preamp = round(-(raw_peak + 0.5), 1)
        print(f"Preamp     : true peak={raw_peak:.3f} dB -> {preamp:+.1f} dB")

    extra_comments = []
    if args.boost_cuts:
        print("Appending boost/cut filters:")
        fcs, Qs, gains, extra_comments = apply_boost_cuts(
            args.boost_cuts, fcs, Qs, gains)
        raw_peak = compute_true_peak(fcs, Qs, gains)
        preamp   = round(-(raw_peak + 0.5), 1)
        print(f"  Preamp after boost/cut: {preamp:+.1f} dB")

    joint_note   = f"joint iters={args.joint_iters}" if args.joint_iters > 0 else "joint=off"
    realloc_note = "realloc=on" if not args.no_realloc else "realloc=off"

    header = "\n".join([
        "# " + "=" * 66,
        f"# AutoEQ v17: {os.path.basename(args.left)} + {os.path.basename(args.right)}",
        f"#           -> {tgt_label}",
        f"# Range     : {f_start:.0f}-{f_end:.0f} Hz | norm @ {args.norm_freq:.0f} Hz [{args.norm_mode}] | delta={norm_offset:+.3f} dB",
        f"# Bands     : {n_opt} optimizer ({warp_label}) + {len(fcs)-n_opt} boost/cut",
        f"# Optim     : biquad IRLS [v17: wclamp+splinecurv+phslope+residsamp+greedyrealloc{'|spawn' if args.spawn else ''}]",
        f"#             + {joint_note} [adaptive fc] + {realloc_note}",
        f"#             lam={args.lam} | lam_s={args.lam_smooth} | lam_e={args.lam_energy}",
        f"#             lam_q={args.lam_q} | lam_gd={args.lam_gd} | lam_driver={args.lam_driver}",
        f"#             minphase={'on' if not args.no_minphase else 'off'}",
        f"# RMSE      : {rmse:.4f} dB (exact biquad) | Max: {max_err:.4f} dB @ {max_err_f:.0f} Hz",
        f"# Preamp    : {preamp:+.1f} dB (true 16k-pt biquad peak = {raw_peak:.3f} dB)",
        "# " + "=" * 66,
    ])

    acc = "\n".join([
        "# " + "-" * 52,
        "# ACCURACY (vs target, boost/cut excluded):",
        *("# " + l for l in accuracy_table(f_fit, pred, c_fit).splitlines()),
        "# " + "-" * 52,
    ])
    peq = "\n".join(["# === PARAMETRIC EQ ===", "",
                     format_peq(fcs, Qs, gains, preamp, n_opt, extra_comments,
                                f_start=f_start, f_end=f_end,
                                gain_floor=args.gain_floor)])
    sections = [header, "", acc, "", peq]
    if args.geq:
        sections += ["", "\n".join(["# === GRAPHIC EQ (correction curve only, no boost/cut) ===", "",
                                    format_geq(fcs[:n_opt], corr_fn)])]

    with open(args.output, "w", encoding="utf-8") as f:
        f.write("\n".join(sections))

    print(f"\nSaved      -> {args.output}")
    print(f"Preamp     : {preamp:+.1f} dB")
    print("\nAccuracy (vs target):")
    print(accuracy_table(f_fit, pred, c_fit))

    print_ir_diagnostic(fcs[:n_opt], Qs[:n_opt], gains[:n_opt])

    if not getattr(args, "no_simulate", False):
        sim_path = args.output.replace(".txt", "_PostEQ.txt")
        simulate_post_eq(l_pts, r_pts, fcs, Qs, gains, preamp, sim_path, fs=args.fs)
        print(f"Post-EQ sim -> {sim_path}  (upload to squig.link for visual check)")


if __name__ == "__main__":
    main()