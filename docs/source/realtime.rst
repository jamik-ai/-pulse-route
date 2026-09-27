Потоковый контур
================

Телеметрия приходит кадрами NDTP по TCP (одно соединение на терминал). Онлайн-движок держит буфер каждого ТС и на
каждой границе 5 минут времени потока строит прогнозы батчем. Момент T «созревает», когда водяной знак потока
(максимальное время − лаг) прошёл T, — поэтому прогноз не использует данных после T.

NDTP
----

.. automodule:: src.realtime.ndtp

Онлайн-движок
-------------

.. automodule:: src.realtime.online

NDTP-сервер
-----------

.. automodule:: src.realtime.server

Проигрывание архива
-------------------

.. automodule:: src.realtime.replay

Проверки на потоке
------------------

Скор Data Science воспроизводится из потока (так его перепроверяют организаторы), а онлайн-прогнозы по всем ТС
сверяются с фактами:

.. code-block:: bash

    ./run.sh python -m src.realtime.stream_submission   # сабмит из потока NDTP
    ./run.sh python -m src.realtime.check_all           # все ТС за день: точность, горизонт, калибровка

.. automodule:: src.realtime.stream_submission

.. automodule:: src.realtime.check_all

.. automodule:: src.realtime.check
