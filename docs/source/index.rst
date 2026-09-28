Пульс маршрута
==============

.. raw:: html

   <div class="hero">
     <p class="lead"><b>«Пульс маршрута» — система раннего предупреждения об опозданиях наземного транспорта
     и дашборд диспетчера</b>, разработанные командой <b>venik</b> для хакатона Московского транспорта.
     Принимает поток телеметрии NDTP, накладывает его на плановое расписание и <b>за 10–15 минут</b>
     до события прогнозирует опоздание на остановке — с вероятностью, интервалом, причиной и рекомендацией
     диспетчеру.</p>
     <div class="stats">
       <div class="stat"><b>0.752</b><span>score лидерборда</span></div>
       <div class="stat"><b>64 с</b><span>MAE на test (baseline 93 с)</span></div>
       <div class="stat"><b>11–15 мин</b><span>упреждение на потоке</span></div>
       <div class="stat"><b>~100 мс</b><span>ML p50 на батч всех ТС</span></div>
     </div>
   </div>

.. image:: ../img/dashboard.jpg
   :class: shot
   :alt: Дашборд диспетчера: карта, карточка события и рекомендации

Разделы
-------

.. raw:: html

   <div class="cards">
     <a class="card" href="overview.html"><b>Обзор</b><span>Задача, как работает система, ключевые цифры</span></a>
     <a class="card" href="architecture.html"><b>Архитектура</b><span>Три модуля в Docker, поток данных, запуск</span></a>
     <a class="card" href="data.html"><b>Данные</b><span>Загрузка датасета, синтетика</span></a>
     <a class="card" href="ml.html"><b>ML-ядро</b><span>Признаки на момент T, ансамбль CatBoost, вероятность, интервал, причины</span></a>
     <a class="card" href="realtime.html"><b>Потоковый контур</b><span>NDTP-кодек, онлайн-движок, replay, проверки на потоке</span></a>
     <a class="card" href="services.html"><b>Сервисы и API</b><span>ML-сервис, backend, маршруты по дорогам, эндпоинты</span></a>
     <a class="card" href="dashboard.html"><b>Дашборд</b><span>Карта, ленты, очередь событий, рекомендации, аналитика</span></a>
     <a class="card" href="quality.html"><b>Качество и надёжность</b><span>Точность, калибровка, латентность, деградация</span></a>
   </div>

Спецификация API — Swagger backend ``/docs`` и ML-сервиса ``:8001/docs``; JSON-файлы — ``docs/openapi_backend.json``,
``docs/openapi_ml.json``. Код: https://github.com/jamik-ai/-pulse-route.

.. toctree::
   :maxdepth: 2
   :hidden:

   overview
   architecture
   data
   ml
   realtime
   services
   dashboard
   quality
