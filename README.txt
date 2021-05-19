Алгоритм эксперимента таков:

1. Обучить агентов на карте 2m2mFOX - smacexp20learn2.py
2. Обучить агентов на карте 2m2mFOXReverse - smacexp20learn2reverse.py
3. Протестировать на карте 2m2mFOX - smacexp21test.py (Q-таблица в se20.pkl)
4. Протестировать на карте 2m2mFOXReverse - smacexp21test2reverse.py (Q-таблица в se21.pkl)
5. Протестировать (!)просуммированную Q-таблицу(!) на карте 2m2mFOX - smacexp21testSummMax.py (Q-таблицы складываются в коде)
6. Протестировать (!)просуммированную Q-таблицу(!) на карте 2m2mFOXReverse - smacexp21testREVSummMax.py (Q-таблицы складываются в коде)
Итоги и анализ.xlsx - данные о проведённых экспериментах от 2020.
