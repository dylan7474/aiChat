#define _GNU_SOURCE
#include <arpa/inet.h>
#include <ctype.h>
#include <errno.h>
#include <netinet/in.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

#include <curl/curl.h>
#include <json-c/json.h>

#define DEFAULT_OLLAMA_URL "http://127.0.0.1:11434/api/generate"
#define SYSTEM_PROMPT                                                                            \
    "You are a helpful and creative AI assistant in a conversation with other friendly AI "   \
    "companions. The user has started the conversation with a topic. Engage in a natural, "    \
    "back-and-forth discussion, building on what the other AI says. Keep your responses "      \
    "concise.\n\n"

#define MAX_PARTICIPANTS 6
#define MAX_NAME_LENGTH 64
#define MAX_MODEL_LENGTH 256
#define MIN_TURNS 1
#define MAX_TURNS 12
#define DEFAULT_PORT 8080
#define READ_BUFFER_CHUNK 4096

struct MemoryStruct {
    char *memory;
    size_t size;
};

struct Participant {
    char name[MAX_NAME_LENGTH];
    char model[MAX_MODEL_LENGTH];
};

static const char *get_ollama_url(void) {
    const char *env = getenv("OLLAMA_URL");
    if (env && *env) {
        return env;
    }
    return DEFAULT_OLLAMA_URL;
}

static size_t WriteMemoryCallback(void *contents, size_t size, size_t nmemb, void *userp) {
    size_t realsize = size * nmemb;
    struct MemoryStruct *mem = (struct MemoryStruct *)userp;
    char *ptr = realloc(mem->memory, mem->size + realsize + 1);
    if (!ptr) {
        fprintf(stderr, "Error: not enough memory (realloc returned NULL)\n");
        return 0;
    }
    mem->memory = ptr;
    memcpy(&(mem->memory[mem->size]), contents, realsize);
    mem->size += realsize;
    mem->memory[mem->size] = '\0';
    return realsize;
}

static char *parse_ollama_response(const char *json_string) {
    struct json_object *parsed_json = NULL;
    struct json_object *response_obj = NULL;
    struct json_object *error_obj = NULL;
    char *response_text = NULL;

    parsed_json = json_tokener_parse(json_string);
    if (!parsed_json) {
        fprintf(stderr, "Error: Could not parse JSON response.\n");
        return NULL;
    }

    if (json_object_object_get_ex(parsed_json, "error", &error_obj)) {
        const char *error_msg = json_object_get_string(error_obj);
        if (error_msg) {
            fprintf(stderr, "Error from AI server: %s\n", error_msg);
        }
    } else if (json_object_object_get_ex(parsed_json, "response", &response_obj)) {
        const char *response_str = json_object_get_string(response_obj);
        if (response_str) {
            response_text = strdup(response_str);
        }
    }

    json_object_put(parsed_json);
    return response_text;
}

static char *get_ai_response(const char *full_prompt, const char *model_name, const char *ollama_url) {
    CURL *curl = NULL;
    char *response = NULL;
    struct MemoryStruct chunk = {.memory = malloc(1), .size = 0};

    if (!chunk.memory) {
        fprintf(stderr, "Failed to allocate memory for response buffer.\n");
        return NULL;
    }

    curl_global_init(CURL_GLOBAL_ALL);
    curl = curl_easy_init();
    if (curl) {
        json_object *jobj = json_object_new_object();
        struct curl_slist *headers = NULL;

        json_object_object_add(jobj, "model", json_object_new_string(model_name));
        json_object_object_add(jobj, "prompt", json_object_new_string(full_prompt));
        json_object_object_add(jobj, "stream", json_object_new_boolean(0));

        const char *json_payload = json_object_to_json_string(jobj);
        headers = curl_slist_append(NULL, "Content-Type: application/json");

        curl_easy_setopt(curl, CURLOPT_URL, ollama_url);
        curl_easy_setopt(curl, CURLOPT_POSTFIELDS, json_payload);
        curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteMemoryCallback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, (void *)&chunk);

        fprintf(stdout, "Requesting response from model '%s'...\n", model_name);
        CURLcode res = curl_easy_perform(curl);
        if (res == CURLE_OK) {
            response = parse_ollama_response(chunk.memory);
        } else {
            fprintf(stderr, "curl_easy_perform() failed: %s\n", curl_easy_strerror(res));
        }

        curl_easy_cleanup(curl);
        curl_slist_free_all(headers);
        json_object_put(jobj);
    }
    curl_global_cleanup();
    free(chunk.memory);
    return response;
}

static char *append_to_history(char *history, const char *text) {
    size_t old_len = history ? strlen(history) : 0;
    size_t text_len = strlen(text);
    char *new_history = realloc(history, old_len + text_len + 1);
    if (!new_history) {
        fprintf(stderr, "Failed to reallocate memory for history.\n");
        free(history);
        return NULL;
    }
    memcpy(new_history + old_len, text, text_len + 1);
    return new_history;
}

static int run_conversation(const char *topic, int turns, struct Participant *participants,
                            size_t participant_count, const char *ollama_url, json_object **out_json,
                            char **error_out) {
    char *conversation_history = NULL;
    json_object *messages = NULL;
    json_object *participants_json = NULL;
    json_object *result = NULL;

    *out_json = NULL;
    if (error_out) {
        *error_out = NULL;
    }

    conversation_history = strdup(SYSTEM_PROMPT);
    if (!conversation_history) {
        if (error_out) {
            *error_out = strdup("Failed to allocate conversation history.");
        }
        return -1;
    }

    conversation_history = append_to_history(conversation_history, "USER: ");
    if (!conversation_history) {
        if (error_out) {
            *error_out = strdup("Failed to build conversation history.");
        }
        return -1;
    }

    conversation_history = append_to_history(conversation_history, topic);
    if (!conversation_history) {
        if (error_out) {
            *error_out = strdup("Failed to build conversation history.");
        }
        return -1;
    }

    messages = json_object_new_array();
    participants_json = json_object_new_array();
    if (!messages || !participants_json) {
        if (error_out) {
            *error_out = strdup("Failed to allocate JSON structures.");
        }
        goto fail;
    }

    for (size_t p = 0; p < participant_count; ++p) {
        json_object *participant_obj = json_object_new_object();
        if (!participant_obj) {
            if (error_out) {
                *error_out = strdup("Failed to allocate participant JSON.");
            }
            goto fail;
        }
        json_object_object_add(participant_obj, "name", json_object_new_string(participants[p].name));
        json_object_object_add(participant_obj, "model", json_object_new_string(participants[p].model));
        json_object_array_add(participants_json, participant_obj);
    }

    for (int turn = 0; turn < turns; ++turn) {
        for (size_t idx = 0; idx < participant_count; ++idx) {
            char label[128];
            char *response = NULL;
            json_object *message = NULL;

            snprintf(label, sizeof(label), "\n\n%s:", participants[idx].name);
            conversation_history = append_to_history(conversation_history, label);
            if (!conversation_history) {
                if (error_out) {
                    *error_out = strdup("Failed to build conversation history.");
                }
                goto fail;
            }

            response = get_ai_response(conversation_history, participants[idx].model, ollama_url);
            if (!response) {
                if (error_out) {
                    char buffer[256];
                    snprintf(buffer, sizeof(buffer), "Model '%.*s' failed to respond.",
                             (int)(sizeof(buffer) - 40), participants[idx].model);
                    *error_out = strdup(buffer);
                }
                goto fail;
            }

            conversation_history = append_to_history(conversation_history, response);
            if (!conversation_history) {
                free(response);
                if (error_out) {
                    *error_out = strdup("Failed to build conversation history.");
                }
                goto fail;
            }

            message = json_object_new_object();
            if (!message) {
                free(response);
                if (error_out) {
                    *error_out = strdup("Failed to allocate message JSON.");
                }
                goto fail;
            }

            json_object_object_add(message, "turn", json_object_new_int(turn + 1));
            json_object_object_add(message, "participantIndex", json_object_new_int((int)idx));
            json_object_object_add(message, "name", json_object_new_string(participants[idx].name));
            json_object_object_add(message, "model", json_object_new_string(participants[idx].model));
            json_object_object_add(message, "text", json_object_new_string(response));
            json_object_array_add(messages, message);

            free(response);
        }
    }

    result = json_object_new_object();
    if (!result) {
        if (error_out) {
            *error_out = strdup("Failed to allocate result JSON.");
        }
        goto fail;
    }

    json_object_object_add(result, "topic", json_object_new_string(topic));
    json_object_object_add(result, "turns", json_object_new_int(turns));
    json_object_object_add(result, "participants", participants_json);
    json_object_object_add(result, "messages", messages);
    json_object_object_add(result, "history", json_object_new_string(conversation_history));

    free(conversation_history);
    *out_json = result;
    return 0;

fail:
    if (messages) {
        json_object_put(messages);
    }
    if (participants_json) {
        json_object_put(participants_json);
    }
    if (result) {
        json_object_put(result);
    }
    free(conversation_history);
    return -1;
}

static void send_http_response(int client_fd, const char *status, const char *content_type,
                               const char *body) {
    char header[512];
    size_t body_length = body ? strlen(body) : 0;
    int header_len = snprintf(header, sizeof(header),
                              "HTTP/1.1 %s\r\n"
                              "Content-Type: %s\r\n"
                              "Content-Length: %zu\r\n"
                              "Access-Control-Allow-Origin: *\r\n"
                              "Connection: close\r\n\r\n",
                              status, content_type, body_length);
    if (header_len < 0 || (size_t)header_len >= sizeof(header)) {
        return;
    }

    send(client_fd, header, (size_t)header_len, 0);
    if (body_length > 0) {
        send(client_fd, body, body_length, 0);
    }
}

static void send_http_error(int client_fd, const char *status, const char *message) {
    json_object *obj = json_object_new_object();
    const char *payload = NULL;

    if (!obj) {
        send_http_response(client_fd, status, "text/plain; charset=UTF-8", message);
        return;
    }

    json_object_object_add(obj, "error", json_object_new_string(message));
    payload = json_object_to_json_string(obj);
    send_http_response(client_fd, status, "application/json", payload);
    json_object_put(obj);
}

static const char *get_html_page(void) {
    return "<!DOCTYPE html>\n"
           "<html lang=\"en\">\n"
           "<head>\n"
           "  <meta charset=\"UTF-8\" />\n"
           "  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\" />\n"
           "  <title>aiChat Arena</title>\n"
           "  <style>\n"
           "    body { font-family: Arial, sans-serif; margin: 2rem; background: #f5f7fb; color: #1f2933; }\n"
           "    h1 { margin-bottom: 0.25rem; }\n"
           "    .card { background: white; border-radius: 12px; padding: 1.5rem; box-shadow: 0 8px 18px rgba(31, 41, 51, 0.08); }\n"
           "    label { display: block; margin-top: 1rem; font-weight: 600; }\n"
           "    input, select { width: 100%; padding: 0.5rem; margin-top: 0.5rem; border-radius: 8px; border: 1px solid #cbd2d9; }\n"
           "    button { margin-top: 1.5rem; padding: 0.75rem 1.5rem; border: none; border-radius: 10px; background: #3b82f6; color: white; font-size: 1rem; cursor: pointer; }\n"
           "    button:hover { background: #2563eb; }\n"
           "    .participants { margin-top: 1rem; }\n"
           "    .participant { border: 1px solid #e4e7eb; padding: 1rem; border-radius: 10px; margin-bottom: 1rem; background: #f9fafb; }\n"
           "    .log { margin-top: 2rem; white-space: pre-wrap; background: white; padding: 1rem; border-radius: 12px; box-shadow: 0 4px 12px rgba(15, 23, 42, 0.08); }\n"
           "    .message { padding: 0.75rem 1rem; border-radius: 8px; margin-bottom: 0.75rem; background: #eff6ff; border-left: 4px solid #3b82f6; }\n"
           "    .message:nth-child(even) { background: #ecfdf5; border-left-color: #10b981; }\n"
           "    .message strong { display: block; margin-bottom: 0.25rem; }\n"
           "    .actions { display: flex; gap: 0.75rem; flex-wrap: wrap; }\n"
           "  </style>\n"
           "</head>\n"
           "<body>\n"
           "  <div class=\"card\">\n"
           "    <h1>aiChat Arena</h1>\n"
           "    <p>Configure friendly AI companions, pick their Ollama models, and watch them chat about your topic.</p>\n"
           "    <label for=\"topic\">Conversation topic</label>\n"
           "    <input id=\"topic\" placeholder=\"Space exploration strategies\" />\n"
           "    <label for=\"turns\">Number of turns</label>\n"
           "    <input id=\"turns\" type=\"number\" min=\"1\" max=\"12\" value=\"3\" />\n"
           "    <div class=\"actions\">\n"
           "      <button id=\"addParticipant\">Add participant</button>\n"
           "      <button id=\"start\">Start conversation</button>\n"
           "    </div>\n"
           "    <div id=\"participants\" class=\"participants\"></div>\n"
           "    <div id=\"status\" style=\"margin-top:1rem; font-weight:600; color:#ef4444;\"></div>\n"
           "  </div>\n"
           "  <div id=\"transcript\" class=\"log\" style=\"display:none;\">\n"
           "    <h2>Conversation transcript</h2>\n"
           "    <div id=\"messages\"></div>\n"
           "  </div>\n"
           "  <script>\n"
           "    const participantsEl = document.getElementById('participants');\n"
           "    const statusEl = document.getElementById('status');\n"
           "    const messagesEl = document.getElementById('messages');\n"
           "    const transcriptEl = document.getElementById('transcript');\n"
           "    function createParticipant(name, model) {\n"
           "      const wrapper = document.createElement('div');\n"
           "      wrapper.className = 'participant';\n"
           "      wrapper.innerHTML = `\n"
           "        <label>Friendly name</label>\n"
           "        <input name=\"name\" placeholder=\"Astra\" value=\"${name || ''}\" />\n"
           "        <label>Ollama model</label>\n"
           "        <input name=\"model\" placeholder=\"llama3:8b\" value=\"${model || ''}\" />\n"
           "        <button type=\"button\" class=\"remove\">Remove</button>\n"
           "      `;\n"
           "      wrapper.querySelector('.remove').addEventListener('click', () => {\n"
           "        participantsEl.removeChild(wrapper);\n"
           "      });\n"
           "      participantsEl.appendChild(wrapper);\n"
           "    }\n"
           "    document.getElementById('addParticipant').addEventListener('click', (event) => {\n"
           "      event.preventDefault();\n"
           "      createParticipant('', '');\n"
           "    });\n"
           "    document.getElementById('start').addEventListener('click', async (event) => {\n"
           "      event.preventDefault();\n"
           "      statusEl.textContent = '';\n"
           "      messagesEl.innerHTML = '';\n"
           "      transcriptEl.style.display = 'none';\n"
           "      const topic = document.getElementById('topic').value.trim();\n"
           "      const turns = parseInt(document.getElementById('turns').value, 10);\n"
           "      const participantDivs = participantsEl.querySelectorAll('.participant');\n"
           "      const participants = [];\n"
           "      participantDivs.forEach((div, index) => {\n"
           "        const name = div.querySelector('input[name=\"name\"]').value.trim();\n"
           "        const model = div.querySelector('input[name=\"model\"]').value.trim();\n"
           "        if (model) {\n"
           "          participants.push({ name: name || `Companion ${index + 1}`, model });\n"
           "        }\n"
           "      });\n"
           "      if (!topic) {\n"
           "        statusEl.textContent = 'Please provide a topic.';\n"
           "        return;\n"
           "      }\n"
           "      if (Number.isNaN(turns) || turns < 1) {\n"
           "        statusEl.textContent = 'Please provide a valid number of turns.';\n"
           "        return;\n"
           "      }\n"
           "      if (participants.length === 0) {\n"
           "        statusEl.textContent = 'Add at least one participant with a model name.';\n"
           "        return;\n"
           "      }\n"
           "      statusEl.textContent = 'Running conversation...';\n"
           "      try {\n"
           "        const response = await fetch('/chat', {\n"
           "          method: 'POST',\n"
           "          headers: { 'Content-Type': 'application/json' },\n"
           "          body: JSON.stringify({ topic, turns, participants })\n"
           "        });\n"
           "        const payload = await response.json();\n"
           "        if (!response.ok) {\n"
           "          statusEl.textContent = payload.error || 'The conversation failed.';\n"
           "          return;\n"
           "        }\n"
           "        statusEl.textContent = '';\n"
           "        transcriptEl.style.display = 'block';\n"
           "        payload.messages.forEach((message) => {\n"
           "          const item = document.createElement('div');\n"
           "          item.className = 'message';\n"
           "          item.innerHTML = `<strong>${message.name} <span style=\"color:#64748b; font-weight:400;\">(${message.model})</span></strong>${message.text}`;\n"
           "          messagesEl.appendChild(item);\n"
           "        });\n"
           "      } catch (error) {\n"
           "        statusEl.textContent = 'Unable to reach the aiChat server.';\n"
           "      }\n"
           "    });\n"
           "    createParticipant('Astra', 'gemma:2b');\n"
           "    createParticipant('Nova', 'llama3:8b');\n"
           "  </script>\n"
           "</body>\n"
           "</html>\n";
}

static int parse_int_header(const char *headers, const char *key) {
    const char *location = strcasestr(headers, key);
    if (!location) {
        return -1;
    }
    location += strlen(key);
    while (*location && isspace((unsigned char)*location)) {
        location++;
    }
    return atoi(location);
}

static int read_http_request(int client_fd, char **out_request, size_t *out_length) {
    size_t capacity = READ_BUFFER_CHUNK;
    size_t length = 0;
    char *buffer = malloc(capacity);
    if (!buffer) {
        return -1;
    }

    while (1) {
        ssize_t bytes = recv(client_fd, buffer + length, capacity - length, 0);
        if (bytes <= 0) {
            free(buffer);
            return -1;
        }
        length += (size_t)bytes;

        char *header_end = memmem(buffer, length, "\r\n\r\n", 4);
        if (header_end) {
            size_t header_length = (size_t)(header_end - buffer) + 4;
            int content_length = parse_int_header(buffer, "Content-Length:");
            size_t total_length = header_length + (content_length > 0 ? (size_t)content_length : 0);
            while (length < total_length) {
                if (length == capacity) {
                    capacity *= 2;
                    char *tmp = realloc(buffer, capacity);
                    if (!tmp) {
                        free(buffer);
                        return -1;
                    }
                    buffer = tmp;
                }
                bytes = recv(client_fd, buffer + length, capacity - length, 0);
                if (bytes <= 0) {
                    free(buffer);
                    return -1;
                }
                length += (size_t)bytes;
            }
            *out_request = buffer;
            *out_length = length;
            return 0;
        }

        if (length == capacity) {
            capacity *= 2;
            char *tmp = realloc(buffer, capacity);
            if (!tmp) {
                free(buffer);
                return -1;
            }
            buffer = tmp;
        }
    }
}

static void handle_chat_request(int client_fd, const char *body, size_t body_length, const char *ollama_url) {
    json_object *payload = NULL;
    json_object *topic_obj = NULL;
    json_object *turns_obj = NULL;
    json_object *participants_obj = NULL;
    const char *topic = NULL;
    int turns = 0;
    struct Participant participants[MAX_PARTICIPANTS];
    size_t participant_count = 0;
    json_object *result = NULL;
    char *error_message = NULL;

    struct json_tokener *tok = json_tokener_new();
    if (!tok) {
        send_http_error(client_fd, "500 Internal Server Error", "Unable to initialise JSON parser.");
        return;
    }

    payload = json_tokener_parse_ex(tok, body, (int)body_length);
    if (json_tokener_get_error(tok) != json_tokener_success || !payload) {
        json_tokener_free(tok);
        send_http_error(client_fd, "400 Bad Request", "Invalid JSON payload.");
        return;
    }
    json_tokener_free(tok);

    if (!json_object_object_get_ex(payload, "topic", &topic_obj) ||
        json_object_get_type(topic_obj) != json_type_string) {
        json_object_put(payload);
        send_http_error(client_fd, "400 Bad Request", "Field 'topic' is required.");
        return;
    }
    topic = json_object_get_string(topic_obj);

    if (!json_object_object_get_ex(payload, "turns", &turns_obj)) {
        json_object_put(payload);
        send_http_error(client_fd, "400 Bad Request", "Field 'turns' is required.");
        return;
    }
    turns = json_object_get_int(turns_obj);
    if (turns < MIN_TURNS) {
        turns = MIN_TURNS;
    }
    if (turns > MAX_TURNS) {
        turns = MAX_TURNS;
    }

    if (!json_object_object_get_ex(payload, "participants", &participants_obj) ||
        json_object_get_type(participants_obj) != json_type_array) {
        json_object_put(payload);
        send_http_error(client_fd, "400 Bad Request", "Field 'participants' must be an array.");
        return;
    }

    size_t array_len = json_object_array_length(participants_obj);
    if (array_len == 0) {
        json_object_put(payload);
        send_http_error(client_fd, "400 Bad Request", "Provide at least one participant.");
        return;
    }
    if (array_len > MAX_PARTICIPANTS) {
        array_len = MAX_PARTICIPANTS;
    }

    for (size_t i = 0; i < array_len; ++i) {
        json_object *item = json_object_array_get_idx(participants_obj, i);
        json_object *name_obj = NULL;
        json_object *model_obj = NULL;
        const char *name = NULL;
        const char *model = NULL;

        if (!item || json_object_get_type(item) != json_type_object) {
            continue;
        }

        if (json_object_object_get_ex(item, "model", &model_obj) &&
            json_object_get_type(model_obj) == json_type_string) {
            model = json_object_get_string(model_obj);
        }
        if (!model || !*model) {
            continue;
        }

        if (json_object_object_get_ex(item, "name", &name_obj) &&
            json_object_get_type(name_obj) == json_type_string && json_object_get_string_len(name_obj) > 0) {
            name = json_object_get_string(name_obj);
        }
        if (!name || !*name) {
            static const char *fallback_names[] = {"Astra", "Nova", "Cosmo", "Lyric", "Echo", "Muse"};
            size_t fallback_idx = participant_count < (sizeof(fallback_names) / sizeof(fallback_names[0]))
                                      ? participant_count
                                      : participant_count % (sizeof(fallback_names) / sizeof(fallback_names[0]));
            name = fallback_names[fallback_idx];
        }

        strncpy(participants[participant_count].name, name, MAX_NAME_LENGTH - 1);
        participants[participant_count].name[MAX_NAME_LENGTH - 1] = '\0';
        strncpy(participants[participant_count].model, model, MAX_MODEL_LENGTH - 1);
        participants[participant_count].model[MAX_MODEL_LENGTH - 1] = '\0';
        participant_count++;
    }

    json_object_put(payload);

    if (participant_count == 0) {
        send_http_error(client_fd, "400 Bad Request", "No valid participants supplied.");
        return;
    }

    if (run_conversation(topic, turns, participants, participant_count, ollama_url, &result, &error_message) != 0) {
        if (error_message) {
            send_http_error(client_fd, "500 Internal Server Error", error_message);
            free(error_message);
        } else {
            send_http_error(client_fd, "500 Internal Server Error", "Conversation failed.");
        }
        return;
    }

    const char *json_payload = json_object_to_json_string_ext(result, JSON_C_TO_STRING_PLAIN);
    send_http_response(client_fd, "200 OK", "application/json", json_payload);
    json_object_put(result);
}

static void handle_client(int client_fd, const char *ollama_url) {
    char *request = NULL;
    size_t request_len = 0;
    char method[8] = {0};
    char path[64] = {0};
    char *body = NULL;
    size_t body_length = 0;

    if (read_http_request(client_fd, &request, &request_len) != 0) {
        send_http_error(client_fd, "400 Bad Request", "Unable to read request.");
        return;
    }

    sscanf(request, "%7s %63s", method, path);

    char *separator = strstr(request, "\r\n\r\n");
    if (separator) {
        body = separator + 4;
        body_length = request_len - (size_t)(body - request);
    }

    if (strcmp(method, "GET") == 0 && strcmp(path, "/") == 0) {
        send_http_response(client_fd, "200 OK", "text/html; charset=UTF-8", get_html_page());
    } else if (strcmp(method, "POST") == 0 && strcmp(path, "/chat") == 0) {
        if (!body) {
            send_http_error(client_fd, "400 Bad Request", "Missing request body.");
        } else {
            handle_chat_request(client_fd, body, body_length, ollama_url);
        }
    } else if (strcmp(method, "OPTIONS") == 0) {
        const char *response =
            "HTTP/1.1 204 No Content\r\n"
            "Access-Control-Allow-Origin: *\r\n"
            "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
            "Access-Control-Allow-Headers: Content-Type\r\n"
            "Connection: close\r\n\r\n";
        send(client_fd, response, strlen(response), 0);
    } else {
        send_http_error(client_fd, "404 Not Found", "Endpoint not found.");
    }

    free(request);
}

int main(void) {
    int server_fd = -1;
    struct sockaddr_in address;
    int opt = 1;
    int port = DEFAULT_PORT;
    int requested_port = DEFAULT_PORT;
    int fallback_used = 0;
    int port_from_env = 0;
    const char *port_env = getenv("AICHAT_PORT");
    const char *ollama_url = get_ollama_url();

    if (port_env && *port_env) {
        char *endptr = NULL;
        long parsed = strtol(port_env, &endptr, 10);
        if (endptr && *endptr == '\0' && parsed > 0 && parsed < 65535) {
            port = (int)parsed;
            port_from_env = 1;
        } else {
            fprintf(stderr, "Warning: invalid AICHAT_PORT '%s', using default %d.\n", port_env, DEFAULT_PORT);
        }
    }
    requested_port = port;

    server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd == -1) {
        perror("socket");
        return EXIT_FAILURE;
    }

    if (setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt)) < 0) {
        perror("setsockopt");
        close(server_fd);
        return EXIT_FAILURE;
    }

    memset(&address, 0, sizeof(address));
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons((uint16_t)port);

    if (bind(server_fd, (struct sockaddr *)&address, sizeof(address)) < 0) {
        if (!port_from_env && errno == EADDRINUSE) {
            fprintf(stderr, "Port %d unavailable, selecting a free port instead.\n", requested_port);
            address.sin_port = htons(0);
            if (bind(server_fd, (struct sockaddr *)&address, sizeof(address)) < 0) {
                perror("bind");
                close(server_fd);
                return EXIT_FAILURE;
            }
            fallback_used = 1;
        } else {
            perror("bind");
            close(server_fd);
            return EXIT_FAILURE;
        }
    }

    socklen_t addrlen = sizeof(address);
    if (getsockname(server_fd, (struct sockaddr *)&address, &addrlen) < 0) {
        perror("getsockname");
        close(server_fd);
        return EXIT_FAILURE;
    }
    port = ntohs(address.sin_port);

    if (listen(server_fd, 10) < 0) {
        perror("listen");
        close(server_fd);
        return EXIT_FAILURE;
    }

    if (fallback_used) {
        printf("Fell back to port %d.\n", port);
    }

    printf("aiChat web server ready on http://127.0.0.1:%d\n", port);
    printf("Using Ollama endpoint: %s\n", ollama_url);

    while (1) {
        int client_fd;
        socklen_t addrlen = sizeof(address);
        client_fd = accept(server_fd, (struct sockaddr *)&address, &addrlen);
        if (client_fd < 0) {
            if (errno == EINTR) {
                continue;
            }
            perror("accept");
            break;
        }

        handle_client(client_fd, ollama_url);
        shutdown(client_fd, SHUT_RDWR);
        close(client_fd);
    }

    close(server_fd);
    return EXIT_SUCCESS;
}
