# Eval results

## 2026-11-22 — Reasoning controls

GPT-6 Luna passed two live Responses routing cases at both Automatic and Low. The cases exercised a regular answer and a flashlight tool call without executing a device action. A separate Low run issued a clarification tool call and finished the reply after its saved tool result was restored. Claude and OpenRouter passed request and tool-result fixtures. Authenticated checks for those two providers are pending.

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

## 2026-11-22T00:06:03Z — Three-provider adapter regression
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
latency ms: p50 1112 p95 1745 max 2498
```

## 2026-11-22T00:42:35Z — V1.5 routing and actuator expansion
```
provider=openai model=gpt-6-luna cases=73
accuracy 71/73 = 97%
  add_calendar_event  1/1
  add_reminder        4/4
  answer              4/4
  calendar_agenda     1/1
  calendar_availabili 1/1
  call_contact        4/4
  clarify             6/6
  complete_reminder   1/1
  create_bear_note    1/1
  create_note         3/3
  directions          5/5
  edit_calendar_event 0/1
  find_messages       1/1
  flashlight          3/3
  list_reminders      1/1
  media_control       6/6
  read_clipboard      1/1
  request_lyft        1/1
  request_ride        2/3
  reschedule_reminder 1/1
  save_item           2/2
  search_saved_items  3/3
  send_message        7/7
  set_alarm           3/3
  set_brightness      1/1
  set_focus           3/3
  set_volume          1/1
  start_timer         4/4
latency ms: p50 1137 p95 2289 max 2645
misses:
  "Move tomorrow's Design Review to 3 PM tomorrow" expected edit_calendar_event got calendar_agenda
  "order a ride to the airport" expected request_ride got clarify (requested another turn)
```

## 2026-11-22T00:44:00Z — V1.5 routing after refining Calendar edit and ride selection
```
provider=openai model=gpt-6-luna cases=73
accuracy 72/73 = 98%
  add_calendar_event  1/1
  add_reminder        4/4
  answer              4/4
  calendar_agenda     1/1
  calendar_availabili 1/1
  call_contact        4/4
  clarify             6/6
  complete_reminder   1/1
  create_bear_note    1/1
  create_note         3/3
  directions          4/5
  edit_calendar_event 1/1
  find_messages       1/1
  flashlight          3/3
  list_reminders      1/1
  media_control       6/6
  read_clipboard      1/1
  request_lyft        1/1
  request_ride        3/3
  reschedule_reminder 1/1
  save_item           2/2
  search_saved_items  3/3
  send_message        7/7
  set_alarm           3/3
  set_brightness      1/1
  set_focus           3/3
  set_volume          1/1
  start_timer         4/4
latency ms: p50 1017 p95 1636 max 5388
misses:
  "how do I get home" expected directions got clarify (requested another turn)
```

## 2026-11-22T00:49:20Z — V1.5 routing with Messages sender and conversation
```
provider=openai model=gpt-6-luna cases=75
accuracy 74/75 = 98%
  add_calendar_event  1/1
  add_reminder        4/4
  answer              4/4
  calendar_agenda     1/1
  calendar_availabili 1/1
  call_contact        4/4
  clarify             6/6
  complete_reminder   1/1
  create_bear_note    1/1
  create_note         2/3
  directions          5/5
  edit_calendar_event 1/1
  find_messages       1/1
  find_recent_message 2/2
  flashlight          3/3
  list_reminders      1/1
  media_control       6/6
  read_clipboard      1/1
  request_lyft        1/1
  request_ride        3/3
  reschedule_reminder 1/1
  save_item           2/2
  search_saved_items  3/3
  send_message        7/7
  set_alarm           3/3
  set_brightness      1/1
  set_focus           3/3
  set_volume          1/1
  start_timer         4/4
latency ms: p50 988 p95 1408 max 1763
misses:
  "create a note idea for the pitch deck lead with the demo" expected create_note got clarify (requested another turn)
```
