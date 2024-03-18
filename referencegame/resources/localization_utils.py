LANGUAGES = {"de", "en", "it", "ja", "pt", "tk", "tr", "zh"}

RESPONSE_PATTERNS = {"de":
                         {"p1":'^ausdruck:\s*(.+)\n*(.+)*$',
                          "p2": '^antwort:\s*(erstes|zweites|drittes)',
                          "p1_tag": 'ausdruck:',
                          "p2_tag": 'antwort:'},
                    "en":
                         {"p1":'^expression:\s*(.+)\n*(.+)*$',
                          "p2": '^answer:\s*(first|second|third)',
                          "p1_tag": 'expression:',
                          "p2_tag": 'answer:'},
                    "it":
                         {"p1":'^espressione:\s*(.+)\n*(.+)*$',
                          "p2": '^risposta:\s*(primo|secondo|terzo)',
                          "p1_tag": 'espressione:',
                          "p2_tag": 'risposta:'},
                    "ja":
                         {"p1":'^表現：\s*(.+)\n*(.+)*$',
                          "p2": '^答：\s*(1番目|2番目|3番目)',
                          "p1_tag": '表現：',
                          "p2_tag": '答：'},
                    "pt":
                         {"p1":'^expressão:\s*(.+)\n*(.+)*$',
                          "p2": '^resposta:\s*(primeiro|segundo|terceiro)',
                          "p1_tag": 'expressão:',
                          "p2_tag": 'resposta:'},
                     "tk":
                         {"p1":'^aňlatma:\s*(.+)\n*(.+)*$',
                          "p2": '^jogap:\s*(birinji|ikinji|üçünji)',
                          "p1_tag": 'aňlatma:',
                          "p2_tag": 'jogap:'},
                    "tr":
                         {"p1":'^ifade:\s*(.+)\n*(.+)*$',
                          "p2": '^cevap:\s*(birinci|ikinci|üçüncü)',
                          "p1_tag": 'ifade:',
                          "p2_tag": 'cevap:'}
,
                    "zh":
                         {"p1":'^表达式:\s*(.+)\n*(.+)*$',
                          "p2": '^回答:\s*(第一|第二|第三)',
                          "p1_tag": '^表达式:',
                          "p2_tag": '回答:'}
                     }
