# Eval results

## 2026-11-21T21:34:43Z — baseline v1 prompt
```
provider=openai model=gpt-5.4-mini cases=52
accuracy 49/52 = 94%
  add_reminder     4/4
  answer           4/4
  call_contact     3/3
  clarify          6/6
  create_note      3/3
  directions       3/4
  flashlight       3/3
  media_control    6/6
  request_ride     3/3
  send_message     5/6
  set_alarm        3/3
  set_focus        3/3
  start_timer      3/4
latency ms: p50 727 p95 1700 max 2342
misses:
  "tell my wife I love her" expected send_message got clarify
  "directions to the airport" expected directions got clarify
  "set a time or for ten minutes" expected start_timer got clarify
```

## 2026-11-21T21:34:56Z — baseline v1 prompt
```
provider=openai model=gpt-5.4-nano cases=52
accuracy 47/52 = 90%
  add_reminder     4/4
  answer           4/4
  call_contact     2/3
  clarify          6/6
  create_note      3/3
  directions       3/4
  flashlight       3/3
  media_control    6/6
  request_ride     2/3
  send_message     5/6
  set_alarm        2/3
  set_focus        3/3
  start_timer      4/4
latency ms: p50 900 p95 1252 max 1332
misses:
  "tell my wife I love her" expected send_message got clarify
  "give Sam a ring" expected call_contact got clarify
  "alarm for quarter to nine tonight" expected set_alarm got clarify
  "directions to the airport" expected directions got clarify
  "order a ride to the airport" expected request_ride got clarify
```

## 2026-11-21T21:35:15Z — prompt v2: don't over-clarify
```
provider=openai model=gpt-5.4-mini cases=52
accuracy 51/52 = 98%
  add_reminder     4/4
  answer           4/4
  call_contact     3/3
  clarify          6/6
  create_note      3/3
  directions       4/4
  flashlight       3/3
  media_control    6/6
  request_ride     3/3
  send_message     6/6
  set_alarm        3/3
  set_focus        3/3
  start_timer      3/4
latency ms: p50 715 p95 980 max 1139
misses:
  "set a time or for ten minutes" expected start_timer got clarify
```

## 2026-11-21T21:35:28Z — prompt v2: don't over-clarify
```
provider=openai model=gpt-5.4-nano cases=52
accuracy 51/52 = 98%
  add_reminder     4/4
  answer           4/4
  call_contact     3/3
  clarify          6/6
  create_note      3/3
  directions       4/4
  flashlight       3/3
  media_control    6/6
  request_ride     2/3
  send_message     6/6
  set_alarm        3/3
  set_focus        3/3
  start_timer      4/4
latency ms: p50 865 p95 1415 max 2174
misses:
  "order a ride to the airport" expected request_ride got clarify
```

## 2026-11-21T23:27:23Z — Device build with per-action clarification prompts
```
provider=openai model=gpt-6-luna cases=61
accuracy 60/61 = 98%
  add_reminder        4/4
  answer              4/4
  call_contact        4/4
  clarify             6/6
  create_bear_note    1/1
  create_note         3/3
  directions          5/5
  flashlight          3/3
  media_control       6/6
  request_ride        3/3
  save_item           2/2
  search_saved_items  3/3
  send_message        7/7
  set_alarm           2/3
  set_focus           3/3
  start_timer         4/4
latency ms: p50 1048 p95 1492 max 2095
misses:
  "wake me up at 7 30" expected set_alarm got clarify (requested another turn)
```

## 2026-11-21T23:44:46Z — App-controlled voice continuation
```
provider=openai model=gpt-6-luna cases=61
accuracy 59/61 = 96%
  add_reminder        4/4
  answer              4/4
  call_contact        4/4
  clarify             6/6
  create_bear_note    1/1
  create_note         2/3
  directions          5/5
  flashlight          3/3
  media_control       6/6
  request_ride        3/3
  save_item           2/2
  search_saved_items  3/3
  send_message        7/7
  set_alarm           2/3
  set_focus           3/3
  start_timer         4/4
latency ms: p50 1066 p95 1528 max 1711
misses:
  "create a note idea for the pitch deck lead with the demo" expected create_note got clarify (requested another turn)
  "wake me up at 7 30" expected set_alarm got clarify (requested another turn)
```

## 2026-11-21T23:56:00Z — Native assistant text and optional tools
```
provider=openai model=gpt-6-luna cases=61
accuracy 61/61 = 100%
  add_reminder        4/4
  answer              4/4
  call_contact        4/4
  clarify             6/6
  create_bear_note    1/1
  create_note         3/3
  directions          5/5
  flashlight          3/3
  media_control       6/6
  request_ride        3/3
  save_item           2/2
  search_saved_items  3/3
  send_message        7/7
  set_alarm           3/3
  set_focus           3/3
  start_timer         4/4
latency ms: p50 1048 p95 1519 max 1841
```
