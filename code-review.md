# Code Review — незакоммиченные изменения (`llm/`)

- **Дата:** 2026-11-21
- **Снапшот:** `git status --porcelain` — 16 изменённых + 6 новых (untracked) файлов, все под `llm/`.
- **Тема изменений:** маппинг имён инструментов на «провод» (`llm/toolname.go` — новый файл). Per-request `toolNameMapping` (`newToolNameMapping` / `wireFor` / `realFor`) введён во всех четырёх провайдерных путях: OpenAI Chat Completions (`provider_openai.go`), OpenAI Responses API (`provider_openai_responses.go`), Anthropic Messages (`provider_anthropic.go`), Google generateContent (`provider_google.go`). Исторический `sanitizeResponsesFunctionName` удалён и заменён общим механизмом; имена MCP-инструментов, нарушающие паттерн `^[a-zA-Z0-9_-]{1,64}$` (точки), транслируются в легальное написание с детерминированным hash-суффиксом при коллизиях; ответы модели обратно транслируются в зарегистрированные имена. Остальные изменённые файлы — тесты: механическая передача `nil`-маппинга в приватные функции и 5 новых тестовых файлов (`toolname_test.go`, `provider_{openai,openai_responses,anthropic,google}_toolname_test.go`).
- **Метод:** полный diff всех 16 изменённых файлов + полные тексты всех 22 файлов; проверка всех call sites `wireFor`/`realFor`; перечисление поверхностей, где имена инструментов вообще сериализуются (definitions, история assistant tool calls / functionCall / functionResponse / tool_use, `tool_choice` — в `llm/` отсутствует, `StreamDelta` — имён не содержит, `tokencount.go` — считает по реальным именам, CountTokens-эндпоинта нет); анализ алгоритма коллизий (детерминизм, завершаемость, бюджет суффикса, UTF-8-границы); консистентность Google-паринга по имени (`callIDToName` + thought-signature ID от реального имени).

## Findings

### MUST FIX

Нет.

### SHOULD FIX

1. **SHOULD FIX** — `googleCompletion` строит `newToolNameMapping` дважды (при построении запроса и при разборе ответа) вместо однократного построения с пробросом.

В `googleCompletion` (`llm/provider_google.go`) маппинг инструментов конструируется дважды внутри одного запроса: строка 195 — inline в вызове `buildGoogleRequest(req, newToolNameMapping(req.Tools, defaultToolNamePolicy, cfg.Logger))`, строка 277 — второй, независимый экземпляр inline в вызове `parseGoogleResponse(req.Model, &genResp, respBody, providerName, newToolNameMapping(req.Tools, defaultToolNamePolicy, cfg.Logger))`. Все три остальных провайдерных пути того же изменения строят маппинг один раз и пробрасывают единственный экземпляр через весь путь: OpenAI Chat Completions (`provider_openai.go:333` — в `buildChatParams` и в обе ветки стрима, включая retry без `stream_options`), Anthropic Messages (`provider_anthropic.go:293` — в `buildRequest` и в стрим), Responses API (`provider_openai_responses.go:119` и `:177` — два отдельных входа, синхронный и стриминговый, каждый строит один раз и передаёт и в `buildResponsesParams`, и в конверсию ответа).

**Root cause:** Google-путь написан в предположении, что `newToolNameMapping` — чистая функция каталога `req.Tools`, поэтому два независимых построения «очевидно» дают одинаковый результат. Этот инвариант нигде не закреплён: сегодня конструкция действительно детерминирована (суффикс коллизии — `toolNameHashSuffix(n, round)`, чистая функция имени и раунда), но это свойство текущей реализации, а не контракта.

**Why it is a problem:** Как только конструкция перестанет быть чистой функцией каталога (в суффикс добавят энтропию — случайность, счётчик, время; tie-break по данным вне каталога; или функция начнёт читать разделяемое состояние), две сборки разойдутся: определения инструментов и история tool-вызовов уйдут в «провод» с правописанием первой сборки, а reverse-lookup при разборе ответа будет переводить имена `functionCall` модели по второй сборке. Реверс-маппинг молча перестанет разрешать имена — без ошибки, tool calls просто вернутся в «чужих» именах. Для Google это особенно чувствительно: он сопаривает `functionResponse` с `functionCall` ПО ИМЕНИ, а не по call ID (см. комментарий в `buildGoogleRequest`). Дополнительные симптомы уже сейчас: при не-nil логгере каждая сборка дублирует Debug-строки о lossy-перезаписях (×2 на запрос), а цикл разрешения коллизий выполняется дважды — лишняя работа на каждый запрос.

**Suggested fix:**
a) Построить маппинг один раз в начале `googleCompletion` (рядом с валидацией `ContentBlocks`): `mapping := newToolNameMapping(req.Tools, defaultToolNamePolicy, cfg.Logger)` — и передать этот единственный экземпляр в `buildGoogleRequest` и в `parseGoogleResponse`; это выровняет Google-путь с тремя остальными провайдерами.
b) Прокинуть маппинг через `googleCompletionConfig` (заполняется вызывающей стороной вместе с остальными полями) — запросная и парсинговая ветки гарантированно используют один экземпляр на уровне сигнатуры; либо вернуть маппинг из `buildGoogleRequest` вместе с `*googleGenerateRequest` и отдать именно его в `parseGoogleResponse`.

## Повторное ревью (свежие независимые проходы)

**Проход 2** (независимый read-only прогон по методике code-review: собственный снапшот `git status --porcelain`, полный diff и полные тексты всех 22 файлов, список находок сформирован до чтения этого файла) нашёл одну находку уровня SHOULD FIX, отсутствовавшую в отчёте: двойное построение `newToolNameMapping` в `googleCompletion` (см. находку №1 выше). Находка внесена; первоначальная пометка «проход новых находок не дал» была преждевременной и исправлена.

**Проход 3** (свежий независимый read-only прогон после внесения находки №1; делегат снял собственный снапшот `git status --porcelain`, прочитал полный diff и полные тексты всех 22 файлов и зафиксировал свой список находок ДО чтения этого файла; интеграционные точки: пути стриминга всех провайдеров, реверс-маппинг в `convertChatResponseMessage` / `parseResponse` / `parseGoogleResponse` / `convertResponsesResponse` / `applyStreamedOutputFallback`, консистентность маппинга между построением запроса и разбором ответа, поведение при nil-каталоге, смена каталога между ходами сессии, кросс-провайдерные сессии, конкурентность — маппинг строится per-request без разделяемого состояния): **новых находок уровней MUST FIX / SHOULD FIX не дал** — единственная находка прохода (двойное построение `newToolNameMapping` в `googleCompletion`) совпадает с уже записанной находкой №1 (локации `provider_google.go:195`/`:277`, root cause, симптомы, вариант фикса a); расхождений в записи находки №1 не обнаружено (строки, ссылки на остальные провайдеры, детерминизм суффикса, pair-by-name Google, отсутствие `tool_choice` в `llm/`, поведение `tokencount.go` — всё подтверждено по коду). Отчёт сошёлся.

## Верификация окружения на момент ревью

- `go vet ./llm/` — чисто.
- `go test ./llm/ -count=1` — `ok github.com/v0lka/sp4rk/llm 29.831s`.
- `golangci-lint run ./llm/...` — `0 issues`.
- `go build ./...` — успешно.

Исходные файлы ревью не изменялись (исправления не вносились — по условию задачи).
