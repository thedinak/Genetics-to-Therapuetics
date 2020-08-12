alternative_drug_names = {'gemcitabine':
                          ['abine',  'accogem',  'acytabin', 'antoril',
                           'axigem', 'bendacitabin', 'biogem',
                           'boligem', 'celzar', 'citegin', 'cytigem',
                           'cytogem', 'daplax', 'dbl', 'demozar', 'dercin',
                           'emcitab', 'enekamub', 'eriogem', 'fotinex',
                           'gebina', 'gemalata', 'gembin', 'gembine', 'gembio',
                           'gemcel', 'gemcetin', 'gemcibine', 'gemcikal',
                           'gemcipen', 'gemcired', 'gemcirena', 'gemcit',
                           'gemcitabin', 'gemcitabina', 'gemcitabine',
                           'gemcitabinum', 'gemcitan', 'gemedac', 'gemflor',
                           'gemful', 'gemita', 'gemko', 'gemliquid', 'gemmis',
                           'gemnil', 'gempower', 'gemsol', 'gemstad',
                           'gemstada', 'gemtabine', 'gemtavis', 'gemtaz',
                           'gemtero', 'gemtra', 'gemtro', 'gemvic', 'gemxit',
                           'gemzar', 'gentabim', 'genuten', 'genvir', 'geroam',
                           'gestredos', 'getanosan', 'getmisi', 'gezt',
                           'gitrabin', 'gramagen', 'haxanit', 'jemta',
                           'kalbezar', 'medigem', 'meditabine', 'nabigem',
                           'nallian', 'oncogem', 'oncoril', 'pamigeno',
                           'ribozar', 'santabin', 'sitagem', 'symtabin',
                           'yu jie', 'ze fei', 'zefei'],
                          'sorafenib': ['nexavar', 'bay-439006'],
                          'doxorubicin': ['adriamycin', 'doxil',
                                          'liposomal doxorubicin'],
                          'doxetaxel': ['taxotere'],
                          'pazonib': ['votrient'],
                          'sunitinib': ['sutent'],
                          'temsirolimus': ['torisel'],
                          'avastin': ['bevacizumab'],
                          'interferon-alpha': ['interferon'],
                          'capecitibine': ['xeloda'],
                          'everolimus': ['afinitor', 'rad001'],
                          'trabectedin': ['et-743'],
                          'gefitinib': ['iressa'],
                          'dacarbazine': ['dtic'],
                          'letrozole': ['femara'],
                          'il-2': ['interleukin', 'interleukin-2', 'il 2',
                                   'il2'],
                          'deforolimus': ['ridaforolimus', 'MK-8669',
                                          'AP23573', 'ap-23573'],
                          'cisplatin': ['platinol'],
                          'carboplatin': ['paraplatin']
                          }

column_set = ['drug_name', 'karnofsky_performance_score',
              'therapy_type', 'vital_status']


kidney_filters_table = {
                        "op": "and",
                        "content": [{"op": "in",
                                    "content": {
                                                "field": "primary_site",
                                                "value": ["Kidney"]}},
                                    {"op": "in",
                                     "content": {
                                                 "field":
                                                 "files.experimental_strategy",
                                                 "value": ["RNA-Seq"]}}]}

fields_table = ["case_id", "primary_site", "project.project_id"]

cases_endpoint = 'https://api.gdc.cancer.gov/cases'
