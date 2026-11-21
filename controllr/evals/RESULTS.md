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
