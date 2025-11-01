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
#define SYSTEM_PROMPT                                                                                 \
    "You are a helpful and creative AI assistant in a conversation with other friendly AI "        \
    "companions. The user has started the conversation with a topic. Engage in a natural, "         \
    "back-and-forth discussion, building on what the other AI says. Keep your responses "           \
    "concise. Speak directly as your assigned participant without narrating the conversation "      \
    "structure, and never reveal your internal thinking—share only your final reply.\n\n"

#define MAX_PARTICIPANTS 6
#define MAX_NAME_LENGTH 64
#define MAX_MODEL_LENGTH 256
#define MIN_TURNS 1
#define MAX_TURNS 12
#define DEFAULT_PORT 17863
#define FALLBACK_PORT_STEPS 3
#define READ_BUFFER_CHUNK 4096

struct MemoryStruct {
    char *memory;
    size_t size;
};

struct Participant {
    char name[MAX_NAME_LENGTH];
    char model[MAX_MODEL_LENGTH];
    char display_model[MAX_MODEL_LENGTH];
};

typedef int (*message_callback)(json_object *message, void *user_data);

static void send_http_response(int client_fd, const char *status, const char *content_type, const char *body);
static void send_http_error(int client_fd, const char *status, const char *message);

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

static void trim_leading_whitespace(char *text) {
    char *start = NULL;

    if (!text) {
        return;
    }

    start = text;
    while (*start && isspace((unsigned char)*start)) {
        start++;
    }
    if (start != text) {
        memmove(text, start, strlen(start) + 1);
    }
}

static void trim_trailing_whitespace(char *text) {
    size_t length = 0;

    if (!text) {
        return;
    }

    length = strlen(text);
    while (length > 0) {
        unsigned char ch = (unsigned char)text[length - 1];
        if (!isspace(ch)) {
            break;
        }
        text[length - 1] = '\0';
        length--;
    }
}

static void remove_tagged_section(char *text, const char *open_tag, const char *close_tag) {
    size_t open_len = 0;
    size_t close_len = 0;

    if (!text || !open_tag || !close_tag) {
        return;
    }

    open_len = strlen(open_tag);
    close_len = strlen(close_tag);
    if (open_len == 0 || close_len == 0) {
        return;
    }

    while (*text) {
        char *start = strcasestr(text, open_tag);
        char *end = NULL;

        if (!start) {
            return;
        }

        end = strcasestr(start + open_len, close_tag);
        if (end) {
            end += close_len;
            memmove(start, end, strlen(end) + 1);
        } else {
            *start = '\0';
            return;
        }
    }
}

static void remove_leading_metadata_block(char *text) {
    static const char *const prefixes[] = {"thought:",         "thinking:",      "thoughts:",
                                           "analysis:",        "reasoning:",     "chain of thought:",
                                           "internal monologue:", "scratchpad:", "plan:"};
    static const char *const markers[] = {"\nanswer:",      "\nfinal answer:", "\nresponse:",
                                          "\nreply:",       "\nfinal:",        "\noutput:",
                                          "\nresult:"};
    char *start = NULL;

    if (!text) {
        return;
    }

    trim_leading_whitespace(text);
    start = text;

    for (size_t i = 0; i < (sizeof(prefixes) / sizeof(prefixes[0])); ++i) {
        size_t prefix_len = strlen(prefixes[i]);
        if (strncasecmp(start, prefixes[i], prefix_len) == 0) {
            char *search_start = start + prefix_len;
            char *removal_end = NULL;

            for (size_t j = 0; j < (sizeof(markers) / sizeof(markers[0])); ++j) {
                char *candidate = strcasestr(search_start, markers[j]);
                if (candidate && (!removal_end || candidate < removal_end)) {
                    removal_end = candidate + 1; /* retain the newline for trimming */
                }
            }

            char *double_newline = strstr(search_start, "\n\n");
            if (double_newline && (!removal_end || double_newline < removal_end)) {
                removal_end = double_newline + 2;
            }

            char *crlf_double = strstr(search_start, "\r\n\r\n");
            if (crlf_double && (!removal_end || crlf_double < removal_end)) {
                removal_end = crlf_double + 4;
            }

            if (removal_end) {
                memmove(start, removal_end, strlen(removal_end) + 1);
            } else {
                *start = '\0';
            }
            break;
        }
    }
}

static void strip_leading_labels(char *text) {
    static const char *const labels[] = {"answer:",      "final answer:", "response:",
                                         "final:",       "reply:",        "output:",
                                         "result:"};
    char *start = NULL;

    if (!text) {
        return;
    }

    trim_leading_whitespace(text);
    start = text;

    for (size_t i = 0; i < (sizeof(labels) / sizeof(labels[0])); ++i) {
        size_t label_len = strlen(labels[i]);
        if (strncasecmp(start, labels[i], label_len) == 0) {
            char *after = start + label_len;
            while (*after && isspace((unsigned char)*after)) {
                after++;
            }
            memmove(start, after, strlen(after) + 1);
            break;
        }
    }
}

static char *find_name_label(char *text, const char *name, char **after_label) {
    size_t name_len = 0;
    char *cursor = NULL;

    if (after_label) {
        *after_label = NULL;
    }

    if (!text || !name || !*name) {
        return NULL;
    }

    name_len = strlen(name);
    cursor = text;

    while ((cursor = strcasestr(cursor, name)) != NULL) {
        char *next = cursor + name_len;

        if (cursor != text) {
            unsigned char prev = (unsigned char)cursor[-1];
            if (isalnum(prev) || prev == '_') {
                cursor += name_len;
                continue;
            }
        }

        while (*next && isspace((unsigned char)*next)) {
            next++;
        }

        if (*next == '(') {
            int depth = 1;
            next++;
            while (*next && depth > 0) {
                if (*next == '(') {
                    depth++;
                } else if (*next == ')') {
                    depth--;
                }
                next++;
            }
            while (*next && isspace((unsigned char)*next)) {
                next++;
            }
        }

        if (*next == ':') {
            char *content = next + 1;
            while (*content && isspace((unsigned char)*content)) {
                content++;
            }
            if (after_label) {
                *after_label = content;
            }
            return cursor;
        }

        cursor += name_len;
    }

    return NULL;
}

static void drop_text_before_name_label(char *text, const char *name) {
    char *after_label = NULL;
    char *label_start = find_name_label(text, name, &after_label);

    if (label_start && label_start != text) {
        memmove(text, label_start, strlen(label_start) + 1);
    }
}

static void strip_leading_name_label(char *text, const char *name) {
    char *after_label = NULL;
    char *label_start = find_name_label(text, name, &after_label);

    if (label_start == text && after_label) {
        memmove(text, after_label, strlen(after_label) + 1);
    }
}

static void sanitize_model_response(char *response, const char *participant_name,
                                    const char *display_label, const char *model_name) {
    const char *labels[3];
    size_t label_count = 0;

    if (!response) {
        return;
    }

    if (participant_name && *participant_name) {
        labels[label_count++] = participant_name;
    }
    if (display_label && *display_label) {
        int already_present = 0;
        for (size_t i = 0; i < label_count; ++i) {
            if (strcasecmp(labels[i], display_label) == 0) {
                already_present = 1;
                break;
            }
        }
        if (!already_present) {
            labels[label_count++] = display_label;
        }
    }
    if (model_name && *model_name) {
        int already_present = 0;
        for (size_t i = 0; i < label_count; ++i) {
            if (strcasecmp(labels[i], model_name) == 0) {
                already_present = 1;
                break;
            }
        }
        if (!already_present) {
            labels[label_count++] = model_name;
        }
    }

    remove_tagged_section(response, "<thinking>", "</thinking>");
    remove_tagged_section(response, "<think>", "</think>");
    remove_tagged_section(response, "<analysis>", "</analysis>");
    remove_tagged_section(response, "<scratchpad>", "</scratchpad>");
    remove_tagged_section(response, "[thinking]", "[/thinking]");
    remove_tagged_section(response, "[think]", "[/think]");
    remove_tagged_section(response, "{thinking}", "{/thinking}");
    remove_tagged_section(response, "{think}", "{/think}");

    trim_leading_whitespace(response);
    for (size_t i = 0; i < label_count; ++i) {
        drop_text_before_name_label(response, labels[i]);
    }
    remove_leading_metadata_block(response);
    trim_leading_whitespace(response);
    for (size_t i = 0; i < label_count; ++i) {
        strip_leading_name_label(response, labels[i]);
        trim_leading_whitespace(response);
    }
    strip_leading_labels(response);
    trim_leading_whitespace(response);
    trim_trailing_whitespace(response);
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

static char *get_ai_response(const char *full_prompt, const char *model_name,
                             const char *participant_name, const char *display_label,
                             const char *ollama_url) {
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
            sanitize_model_response(response, participant_name, display_label, model_name);
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

static char *build_models_url(const char *ollama_url) {
    const char *suffix = "/tags";
    const char *generate = "generate";
    size_t url_len = strlen(ollama_url);
    size_t generate_len = strlen(generate);
    char *result = NULL;

    if (url_len >= generate_len && strcmp(ollama_url + url_len - generate_len, generate) == 0) {
        size_t base_len = url_len - generate_len;
        result = malloc(base_len + strlen(suffix) + 1);
        if (!result) {
            return NULL;
        }
        memcpy(result, ollama_url, base_len);
        if (base_len > 0 && result[base_len - 1] == '/') {
            strcpy(result + base_len, suffix + 1);
        } else {
            strcpy(result + base_len, suffix);
        }
        return result;
    }

    int needs_slash = (url_len == 0 || ollama_url[url_len - 1] != '/');
    result = malloc(url_len + needs_slash + strlen(suffix) + 1);
    if (!result) {
        return NULL;
    }
    strcpy(result, ollama_url);
    if (needs_slash) {
        strcat(result, "/");
    }
    strcat(result, suffix + 1); /* suffix starts with '/', avoid duplicating */
    return result;
}

static int fetch_available_models(const char *ollama_url, json_object **out_json, char **error_out) {
    struct MemoryStruct chunk = {.memory = NULL, .size = 0};
    CURL *curl = NULL;
    CURLcode res = CURLE_OK;
    char *models_url = NULL;
    json_object *parsed = NULL;
    json_object *models_array = NULL;
    json_object *result = NULL;
    json_object *list = NULL;

    *out_json = NULL;
    if (error_out) {
        *error_out = NULL;
    }

    models_url = build_models_url(ollama_url);
    if (!models_url) {
        if (error_out) {
            *error_out = strdup("Failed to prepare Ollama models URL.");
        }
        return -1;
    }

    chunk.memory = malloc(1);
    if (!chunk.memory) {
        free(models_url);
        if (error_out) {
            *error_out = strdup("Failed to allocate response buffer.");
        }
        return -1;
    }
    chunk.size = 0;

    curl_global_init(CURL_GLOBAL_ALL);
    curl = curl_easy_init();
    if (!curl) {
        free(models_url);
        free(chunk.memory);
        if (error_out) {
            *error_out = strdup("Unable to initialise CURL.");
        }
        curl_global_cleanup();
        return -1;
    }

    curl_easy_setopt(curl, CURLOPT_URL, models_url);
    curl_easy_setopt(curl, CURLOPT_HTTPGET, 1L);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteMemoryCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, (void *)&chunk);

    res = curl_easy_perform(curl);
    free(models_url);
    curl_easy_cleanup(curl);
    curl_global_cleanup();

    if (res != CURLE_OK) {
        free(chunk.memory);
        if (error_out) {
            *error_out = strdup("Failed to contact Ollama for model list.");
        }
        return -1;
    }

    parsed = json_tokener_parse(chunk.memory);
    if (parsed && json_object_is_type(parsed, json_type_object)) {
        json_object_object_get_ex(parsed, "models", &models_array);
    } else if (parsed && json_object_is_type(parsed, json_type_array)) {
        models_array = parsed;
    }

    if (!models_array || !json_object_is_type(models_array, json_type_array)) {
        free(chunk.memory);
        if (parsed) {
            json_object_put(parsed);
        }
        if (error_out) {
            *error_out = strdup("Unexpected response from Ollama while listing models.");
        }
        return -1;
    }

    list = json_object_new_array();
    if (!list) {
        free(chunk.memory);
        json_object_put(parsed);
        if (error_out) {
            *error_out = strdup("Failed to allocate models array.");
        }
        return -1;
    }

    size_t array_len = json_object_array_length(models_array);
    for (size_t i = 0; i < array_len; ++i) {
        json_object *item = json_object_array_get_idx(models_array, i);
        const char *model_value = NULL;
        const char *name_value = NULL;

        if (!item) {
            continue;
        }

        if (json_object_is_type(item, json_type_object)) {
            json_object *field = NULL;
            if (json_object_object_get_ex(item, "model", &field) && field) {
                model_value = json_object_get_string(field);
            }
            if (json_object_object_get_ex(item, "name", &field) && field) {
                name_value = json_object_get_string(field);
            }
        } else if (json_object_is_type(item, json_type_string)) {
            model_value = json_object_get_string(item);
        }

        if (!model_value || !*model_value) {
            if (name_value && *name_value) {
                model_value = name_value;
            } else {
                continue;
            }
        }

        json_object *entry = json_object_new_object();
        if (!entry) {
            json_object_put(list);
            json_object_put(parsed);
            free(chunk.memory);
            if (error_out) {
                *error_out = strdup("Failed to allocate model entry.");
            }
            return -1;
        }

        const char *display = (name_value && *name_value) ? name_value : model_value;
        json_object_object_add(entry, "name", json_object_new_string(display));
        json_object_object_add(entry, "model", json_object_new_string(model_value));
        json_object_array_add(list, entry);
    }

    result = json_object_new_object();
    if (!result) {
        json_object_put(list);
        json_object_put(parsed);
        free(chunk.memory);
        if (error_out) {
            *error_out = strdup("Failed to prepare models payload.");
        }
        return -1;
    }

    json_object_object_add(result, "models", list);
    *out_json = result;

    json_object_put(parsed);
    free(chunk.memory);
    return 0;
}

static void handle_models_request(int client_fd, const char *ollama_url) {
    json_object *payload = NULL;
    char *error_message = NULL;

    if (fetch_available_models(ollama_url, &payload, &error_message) == 0 && payload) {
        const char *json_payload = json_object_to_json_string_ext(payload, JSON_C_TO_STRING_PLAIN);
        send_http_response(client_fd, "200 OK", "application/json", json_payload);
        json_object_put(payload);
    } else {
        const char *message = error_message ? error_message : "Unable to retrieve model list.";
        send_http_error(client_fd, "502 Bad Gateway", message);
    }

    if (error_message) {
        free(error_message);
    }
}

static const char *lookup_display_model(json_object *models_array, const char *identifier) {
    size_t array_len = 0;

    if (!models_array || !identifier || !*identifier) {
        return NULL;
    }

    if (!json_object_is_type(models_array, json_type_array)) {
        return NULL;
    }

    array_len = json_object_array_length(models_array);
    for (size_t i = 0; i < array_len; ++i) {
        json_object *item = json_object_array_get_idx(models_array, i);
        const char *model_value = NULL;
        const char *name_value = NULL;

        if (!item || !json_object_is_type(item, json_type_object)) {
            continue;
        }

        json_object *field = NULL;
        if (json_object_object_get_ex(item, "model", &field) && field) {
            model_value = json_object_get_string(field);
        }
        if (json_object_object_get_ex(item, "name", &field) && field) {
            name_value = json_object_get_string(field);
        }

        if (model_value && strcmp(model_value, identifier) == 0) {
            if (name_value && *name_value) {
                return name_value;
            }
            return model_value;
        }
        if (name_value && strcmp(name_value, identifier) == 0) {
            return name_value;
        }
    }

    return NULL;
}

static void ensure_participant_display_models(struct Participant *participants, size_t participant_count,
                                              json_object *models_payload) {
    json_object *models_array = NULL;

    if (!participants || participant_count == 0) {
        return;
    }

    if (models_payload && json_object_is_type(models_payload, json_type_object)) {
        json_object_object_get_ex(models_payload, "models", &models_array);
    }

    for (size_t i = 0; i < participant_count; ++i) {
        trim_leading_whitespace(participants[i].display_model);
        trim_trailing_whitespace(participants[i].display_model);
        if (participants[i].display_model[0] != '\0') {
            continue;
        }

        const char *display = lookup_display_model(models_array, participants[i].model);
        if (!display || !*display) {
            display = participants[i].model;
        }

        strncpy(participants[i].display_model, display, MAX_MODEL_LENGTH - 1);
        participants[i].display_model[MAX_MODEL_LENGTH - 1] = '\0';
        trim_leading_whitespace(participants[i].display_model);
        trim_trailing_whitespace(participants[i].display_model);
    }
}

static int run_conversation(const char *topic, int turns, struct Participant *participants,
                            size_t participant_count, const char *ollama_url, message_callback on_message,
                            void *callback_data, json_object **out_json, char **error_out) {
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
        if (participants[p].display_model[0] != '\0') {
            json_object_object_add(participant_obj, "displayModel",
                                   json_object_new_string(participants[p].display_model));
        }
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

            response = get_ai_response(conversation_history, participants[idx].model,
                                       participants[idx].name, participants[idx].display_model,
                                       ollama_url);
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
            if (participants[idx].display_model[0] != '\0') {
                json_object_object_add(message, "displayModel",
                                       json_object_new_string(participants[idx].display_model));
            }
            json_object_object_add(message, "text", json_object_new_string(response));
            json_object_array_add(messages, message);

            if (on_message) {
                json_object_get(message);
                if (on_message(message, callback_data) != 0) {
                    json_object_put(message);
                    free(response);
                    if (error_out && (!*error_out)) {
                        *error_out = strdup("Failed to stream message.");
                    }
                    goto fail;
                }
                json_object_put(message);
            }

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

static int send_all(int client_fd, const char *data, size_t length) {
    size_t total_sent = 0;

    while (total_sent < length) {
        ssize_t written = send(client_fd, data + total_sent, length - total_sent, 0);
        if (written <= 0) {
            return -1;
        }
        total_sent += (size_t)written;
    }

    return 0;
}

static int send_chunked_header(int client_fd, const char *status, const char *content_type) {
    char header[512];
    int header_len = snprintf(header, sizeof(header),
                              "HTTP/1.1 %s\r\n"
                              "Content-Type: %s\r\n"
                              "Transfer-Encoding: chunked\r\n"
                              "Cache-Control: no-cache\r\n"
                              "Access-Control-Allow-Origin: *\r\n"
                              "Connection: close\r\n\r\n",
                              status, content_type);

    if (header_len < 0 || (size_t)header_len >= sizeof(header)) {
        return -1;
    }

    return send_all(client_fd, header, (size_t)header_len);
}

static int send_json_chunk(int client_fd, json_object *obj) {
    const char *json = json_object_to_json_string_ext(obj, JSON_C_TO_STRING_PLAIN);
    char size_buffer[32];
    int size_len = 0;
    size_t json_len = 0;

    if (!json) {
        return -1;
    }

    json_len = strlen(json);
    size_len = snprintf(size_buffer, sizeof(size_buffer), "%zx\r\n", json_len + 1);
    if (size_len < 0 || (size_t)size_len >= sizeof(size_buffer)) {
        return -1;
    }

    if (send_all(client_fd, size_buffer, (size_t)size_len) != 0) {
        return -1;
    }

    if (send_all(client_fd, json, json_len) != 0) {
        return -1;
    }

    if (send_all(client_fd, "\n", 1) != 0) {
        return -1;
    }

    if (send_all(client_fd, "\r\n", 2) != 0) {
        return -1;
    }

    return 0;
}

static int finish_chunked_response(int client_fd) {
    return send_all(client_fd, "0\r\n\r\n", 5);
}

static int send_stream_error_event(int client_fd, const char *message) {
    json_object *event = json_object_new_object();
    int result = -1;

    if (!event) {
        return -1;
    }

    json_object_object_add(event, "type", json_object_new_string("error"));
    json_object_object_add(event, "message",
                           json_object_new_string(message ? message : "Conversation failed."));

    result = send_json_chunk(client_fd, event);
    json_object_put(event);
    return result;
}

struct StreamContext {
    int client_fd;
    int failed;
};

static int stream_message_callback(json_object *message, void *user_data) {
    struct StreamContext *ctx = (struct StreamContext *)user_data;
    json_object *event = NULL;
    int rc = -1;

    if (!ctx || ctx->failed) {
        return -1;
    }

    event = json_object_new_object();
    if (!event) {
        ctx->failed = 1;
        return -1;
    }

    json_object_object_add(event, "type", json_object_new_string("message"));
    json_object_object_add(event, "message", json_object_get(message));

    rc = send_json_chunk(ctx->client_fd, event);
    if (rc != 0) {
        ctx->failed = 1;
    }

    json_object_put(event);
    return rc;
}

static const char *get_html_page(void) {
    return "<!DOCTYPE html>\n"
           "<html lang=\"en\">\n"
           "<head>\n"
           "  <meta charset=\"UTF-8\" />\n"
           "  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\" />\n"
           "  <title>aiChat Arena</title>\n"
           "  <style>\n"
           "    :root { color-scheme: dark; }\n"
           "    * { box-sizing: border-box; }\n"
           "    body { margin: 0; padding: clamp(1.5rem, 3vw, 3rem); font-family: 'Segoe UI', 'Orbitron', 'Roboto', sans-serif; background: radial-gradient(circle at 20% 20%, rgba(56, 189, 248, 0.15), transparent 55%), radial-gradient(circle at 80% 0%, rgba(168, 85, 247, 0.12), transparent 45%), #020617; color: #e2e8f0; min-height: 100vh; display: flex; flex-direction: column; align-items: center; gap: 2.5rem; position: relative; overflow-x: hidden; }\n"
           "    body::before { content: ''; position: fixed; inset: -20vmax; background: conic-gradient(from 180deg at 50% 50%, rgba(56, 189, 248, 0.08), rgba(59, 130, 246, 0.16), rgba(168, 85, 247, 0.12), rgba(56, 189, 248, 0.08)); opacity: 0.65; filter: blur(120px); animation: auroraSpin 48s linear infinite; z-index: -2; pointer-events: none; }\n"
           "    body::after { content: ''; position: fixed; inset: -18vmax; background: radial-gradient(circle at 15% 25%, rgba(59, 130, 246, 0.22), transparent 55%), radial-gradient(circle at 85% 15%, rgba(168, 85, 247, 0.18), transparent 60%); opacity: 0.45; filter: blur(90px); animation: auroraPulse 32s ease-in-out infinite alternate; z-index: -3; pointer-events: none; }\n"
           "    @keyframes auroraSpin {\n"
           "      from { transform: rotate(0deg); }\n"
           "      to { transform: rotate(360deg); }\n"
           "    }\n"
           "    @keyframes auroraPulse {\n"
           "      0% { opacity: 0.35; transform: scale(0.95); }\n"
           "      50% { opacity: 0.55; transform: scale(1.05); }\n"
           "      100% { opacity: 0.35; transform: scale(1); }\n"
           "    }\n"
           "    .card { position: relative; width: min(960px, 100%); background: rgba(15, 23, 42, 0.75); border-radius: 24px; padding: clamp(1.5rem, 3vw, 2.5rem); border: 1px solid rgba(148, 163, 184, 0.25); box-shadow: 0 40px 80px rgba(2, 6, 23, 0.6); backdrop-filter: blur(18px); overflow: hidden; transform-style: preserve-3d; transition: transform 0.6s cubic-bezier(0.22, 1, 0.36, 1), box-shadow 0.6s ease; }\n"
           "    .card::after { content: ''; position: absolute; inset: -40%; background: radial-gradient(circle at 30% 30%, rgba(56, 189, 248, 0.45), transparent 65%), radial-gradient(circle at 70% 10%, rgba(168, 85, 247, 0.35), transparent 60%); opacity: 0; transform: translate3d(0, 40px, 0) scale(0.95); filter: blur(40px); transition: opacity 0.6s ease, transform 0.6s cubic-bezier(0.22, 1, 0.36, 1); pointer-events: none; z-index: 0; }\n"
           "    .card > * { position: relative; z-index: 1; }\n"
           "    .card:hover { transform: translateY(-6px); box-shadow: 0 48px 120px rgba(2, 6, 23, 0.75); }\n"
           "    .card:hover::after { opacity: 1; transform: translate3d(0, 0, 0) scale(1.05); }\n"
           "    h1 { margin-bottom: 0.25rem; font-size: clamp(1.9rem, 3vw, 2.4rem); letter-spacing: 0.08em; text-transform: uppercase; color: #f8fafc; }\n"
           "    h2 { margin-top: 0; letter-spacing: 0.06em; text-transform: uppercase; color: #cbd5f5; }\n"
           "    p { margin-top: 0; color: rgba(226, 232, 240, 0.85); }\n"
           "    label { display: block; margin-top: 1.25rem; font-weight: 600; letter-spacing: 0.05em; text-transform: uppercase; color: rgba(148, 163, 184, 0.9); }\n"
           "    input, select { width: 100%; padding: 0.75rem 1rem; margin-top: 0.5rem; border-radius: 12px; border: 1px solid rgba(148, 163, 184, 0.25); background: rgba(15, 23, 42, 0.6); color: #f8fafc; box-shadow: inset 0 0 0 rgba(15, 23, 42, 0.5); transition: border-color 0.2s ease, box-shadow 0.2s ease, transform 0.3s ease; }\n"
           "    input:focus, select:focus { outline: none; border-color: rgba(94, 234, 212, 0.8); box-shadow: 0 0 0 3px rgba(94, 234, 212, 0.2); transform: translateY(-1px); }\n"
           "    input::placeholder { color: rgba(148, 163, 184, 0.6); }\n"
           "    .actions { display: flex; gap: 0.75rem; flex-wrap: wrap; margin-top: 1.5rem; }\n"
           "    button { position: relative; padding: 0.85rem 1.8rem; border: none; border-radius: 999px; background: linear-gradient(120deg, #22d3ee, #a855f7); background-size: 220% 220%; color: #0b1120; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; cursor: pointer; transition: transform 0.25s ease, box-shadow 0.25s ease, background-position 0.4s ease; box-shadow: 0 12px 26px rgba(168, 85, 247, 0.35); }\n"
           "    button::after { content: ''; position: absolute; inset: -2px; border-radius: inherit; border: 1px solid rgba(255, 255, 255, 0.25); opacity: 0; transition: opacity 0.3s ease, transform 0.3s ease; pointer-events: none; }\n"
           "    button:hover { transform: translateY(-2px); box-shadow: 0 20px 45px rgba(168, 85, 247, 0.45); background-position: 100% 50%; }\n"
           "    button:hover::after { opacity: 1; transform: scale(1.03); }\n"
           "    button:focus-visible { outline: none; box-shadow: 0 0 0 3px rgba(14, 165, 233, 0.35); }\n"
           "    button:active { transform: translateY(0); }\n"
           "    .participants { margin-top: 1.5rem; display: grid; gap: 1rem; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); perspective: 1200px; }\n"
           "    .participant { position: relative; border-radius: 18px; padding: 1.25rem; border: 1px solid var(--participant-border, rgba(148, 163, 184, 0.45)); background: var(--participant-surface, rgba(15, 23, 42, 0.6)); color: #e2e8f0; box-shadow: 0 0 32px var(--participant-glow, rgba(15, 23, 42, 0.8)); overflow: hidden; backdrop-filter: blur(12px); transform-style: preserve-3d; background-size: 180% 180%; transition: transform 0.6s cubic-bezier(0.22, 1, 0.36, 1), box-shadow 0.6s ease, border-color 0.3s ease; }\n"
           "    .participant::before { content: ''; position: absolute; inset: 0; background: var(--participant-pattern, rgba(94, 234, 212, 0.1)); opacity: 0.55; filter: blur(60px); z-index: 0; transition: opacity 0.6s ease, transform 0.6s ease; animation: participantDrift 24s linear infinite; }\n"
           "    .participant::after { content: ''; position: absolute; inset: 1px; border-radius: 16px; background: radial-gradient(circle at 20% 20%, var(--participant-highlight, rgba(255, 255, 255, 0.35)), transparent 55%); opacity: 0; mix-blend-mode: screen; transition: opacity 0.6s ease; z-index: 0; }\n"
           "    .participant > * { position: relative; z-index: 1; }\n"
           "    .participant:hover, .participant:focus-within { transform: translateY(-8px) rotate3d(1, -1, 0, 6deg); box-shadow: 0 32px 80px var(--participant-glow, rgba(15, 23, 42, 0.8)); }\n"
           "    .participant:hover::before, .participant:focus-within::before { opacity: 0.8; transform: scale(1.08); }\n"
           "    .participant:hover::after, .participant:focus-within::after { opacity: 0.85; }\n"
           "    @keyframes participantDrift {\n"
           "      0% { transform: scale(1) translate3d(0, 0, 0); }\n"
           "      50% { transform: scale(1.05) translate3d(-6px, 4px, 0); }\n"
           "      100% { transform: scale(1) translate3d(0, 0, 0); }\n"
           "    }\n"
           "    .participant button { margin-top: 1rem; width: fit-content; background: rgba(15, 23, 42, 0.65); color: #f8fafc; border: 1px solid rgba(148, 163, 184, 0.35); border-radius: 12px; padding: 0.5rem 1rem; letter-spacing: 0.05em; text-transform: none; font-size: 0.85rem; transition: transform 0.3s ease, border-color 0.3s ease, box-shadow 0.3s ease; }\n"
           "    .participant button::after { display: none; }\n"
           "    .participant button:hover, .participant button:focus-visible { color: #f87171; border-color: rgba(248, 113, 113, 0.8); box-shadow: 0 0 20px rgba(248, 113, 113, 0.25); transform: translateY(-1px); }\n"
           "    #status { margin-top: 1rem; font-weight: 600; color: #f97316; letter-spacing: 0.05em; opacity: 0; transform: translateY(-0.25rem); transition: opacity 0.35s ease, transform 0.35s ease; text-shadow: none; }\n"
           "    #status.status-active { opacity: 1; transform: translateY(0); }\n"
           "    #status.status-flash { animation: statusGlow 1.2s ease-out; }\n"
           "    @keyframes statusGlow {\n"
           "      0% { opacity: 0.6; text-shadow: 0 0 0 rgba(249, 115, 22, 0.75); }\n"
           "      45% { opacity: 1; text-shadow: 0 0 22px rgba(249, 115, 22, 0.85); }\n"
           "      100% { opacity: 1; text-shadow: none; }\n"
           "    }\n"
           "    #status:empty { display: none; }\n"
           "    #transcript { width: min(960px, 100%); }\n"
           "    .log { white-space: pre-wrap; background: rgba(15, 23, 42, 0.78); padding: clamp(1.25rem, 3vw, 2rem); border-radius: 24px; border: 1px solid rgba(148, 163, 184, 0.25); box-shadow: 0 24px 60px rgba(2, 6, 23, 0.65); backdrop-filter: blur(16px); }\n"
           "    .message { padding: 1rem 1.25rem; border-radius: 16px; margin-bottom: 0.85rem; background: var(--message-bg, rgba(56, 189, 248, 0.12)); border-left: 4px solid var(--message-border, #38bdf8); box-shadow: 0 12px 28px var(--message-glow, rgba(15, 23, 42, 0.6)); color: #f8fafc; opacity: 0; transform: translateY(16px); transition: transform 0.55s cubic-bezier(0.23, 1, 0.32, 1), opacity 0.55s ease, box-shadow 0.55s ease; }\n"
           "    .message.is-visible { opacity: 1; transform: translateY(0); }\n"
           "    .message.message-summary { background: linear-gradient(135deg, rgba(13, 148, 136, 0.28), rgba(14, 165, 233, 0.3)); border-left-color: #06b6d4; font-style: italic; }\n"
           "    .message.message-summary strong { letter-spacing: 0.12em; color: #bae6fd; }\n"
           "    .message.message-enter { animation: messagePulse 1.2s ease-out forwards; }\n"
           "    @keyframes messagePulse {\n"
           "      0% { box-shadow: 0 0 0 0 var(--message-glow, rgba(56, 189, 248, 0.35)); }\n"
           "      55% { box-shadow: 0 0 36px var(--message-glow, rgba(56, 189, 248, 0.55)); }\n"
           "      100% { box-shadow: 0 12px 28px var(--message-glow, rgba(15, 23, 42, 0.6)); }\n"
           "    }\n"
           "    .message strong { display: block; margin-bottom: 0.35rem; letter-spacing: 0.05em; text-transform: uppercase; color: rgba(226, 232, 240, 0.92); }\n"
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
           "    <div id=\"status\"></div>\n"
           "  </div>\n"
           "  <div id=\"transcript\" class=\"log\" style=\"display:none;\">\n"
           "    <h2>Conversation transcript</h2>\n"
           "    <div id=\"messages\"></div>\n"
           "  </div>\n"
           "  <script>\n"
           "    const participantsEl = document.getElementById('participants');\n"
           "    const statusEl = document.getElementById('status');\n"
           "    statusEl.addEventListener('animationend', (event) => {\n"
           "      if (event.animationName === 'statusGlow') {\n"
           "        statusEl.classList.remove('status-flash');\n"
           "      }\n"
           "    });\n"
           "    function setStatus(message) {\n"
           "      const text = typeof message === 'string' ? message : (message ? String(message) : '');\n"
           "      const active = Boolean(text);\n"
           "      statusEl.textContent = active ? text : '';\n"
           "      statusEl.classList.toggle('status-active', active);\n"
           "      if (active) {\n"
           "        statusEl.classList.remove('status-flash');\n"
           "        void statusEl.offsetWidth;\n"
           "        statusEl.classList.add('status-flash');\n"
           "      } else {\n"
           "        statusEl.classList.remove('status-flash');\n"
           "      }\n"
           "    }\n"
           "    const messagesEl = document.getElementById('messages');\n"
           "    const transcriptEl = document.getElementById('transcript');\n"
           "    const transcriptMessages = [];\n"
           "    let summaryAppended = false;\n"
           "    let currentTopic = '';\n"
           "    let currentTurns = 0;\n"
           "    let currentParticipants = [];\n"
           "    let availableModels = [];\n"
           "    const modelSelects = new Set();\n"
           "    let modelLoadError = false;\n"
           "    let missingModelWarning = false;\n"
           "    const basePalette = [\n"
           "      {\n"
           "        messageBackground: 'linear-gradient(135deg, rgba(56, 189, 248, 0.18), rgba(59, 130, 246, 0.45))',\n"
           "        border: '#38bdf8',\n"
           "        glow: 'rgba(56, 189, 248, 0.45)',\n"
           "        cardBackground: 'linear-gradient(160deg, rgba(12, 74, 110, 0.85), rgba(37, 99, 235, 0.75))',\n"
           "        cardPattern: 'radial-gradient(circle at 20% 20%, rgba(14, 165, 233, 0.35) 0, transparent 45%), radial-gradient(circle at 80% 0%, rgba(59, 130, 246, 0.3) 0, transparent 40%)'\n"
           "      },\n"
           "      {\n"
           "        messageBackground: 'linear-gradient(135deg, rgba(244, 114, 182, 0.2), rgba(236, 72, 153, 0.45))',\n"
           "        border: '#f472b6',\n"
           "        glow: 'rgba(244, 114, 182, 0.45)',\n"
           "        cardBackground: 'linear-gradient(160deg, rgba(88, 28, 135, 0.82), rgba(162, 28, 175, 0.78))',\n"
           "        cardPattern: 'radial-gradient(circle at 25% 20%, rgba(249, 168, 212, 0.32) 0, transparent 42%), radial-gradient(circle at 80% 10%, rgba(236, 72, 153, 0.28) 0, transparent 38%)'\n"
           "      },\n"
           "      {\n"
           "        messageBackground: 'linear-gradient(135deg, rgba(52, 211, 153, 0.2), rgba(16, 185, 129, 0.45))',\n"
           "        border: '#34d399',\n"
           "        glow: 'rgba(16, 185, 129, 0.4)',\n"
           "        cardBackground: 'linear-gradient(160deg, rgba(4, 47, 46, 0.85), rgba(13, 148, 136, 0.78))',\n"
           "        cardPattern: 'radial-gradient(circle at 18% 22%, rgba(94, 234, 212, 0.32) 0, transparent 45%), radial-gradient(circle at 82% 12%, rgba(16, 185, 129, 0.3) 0, transparent 40%)'\n"
           "      },\n"
           "      {\n"
           "        messageBackground: 'linear-gradient(135deg, rgba(251, 191, 36, 0.2), rgba(249, 115, 22, 0.45))',\n"
           "        border: '#f59e0b',\n"
           "        glow: 'rgba(251, 146, 60, 0.45)',\n"
           "        cardBackground: 'linear-gradient(160deg, rgba(88, 40, 12, 0.85), rgba(234, 88, 12, 0.75))',\n"
           "        cardPattern: 'radial-gradient(circle at 24% 16%, rgba(251, 191, 36, 0.32) 0, transparent 44%), radial-gradient(circle at 78% 10%, rgba(249, 115, 22, 0.28) 0, transparent 40%)'\n"
           "      },\n"
           "      {\n"
           "        messageBackground: 'linear-gradient(135deg, rgba(129, 140, 248, 0.2), rgba(99, 102, 241, 0.45))',\n"
           "        border: '#818cf8',\n"
           "        glow: 'rgba(129, 140, 248, 0.45)',\n"
           "        cardBackground: 'linear-gradient(160deg, rgba(30, 41, 102, 0.85), rgba(76, 29, 149, 0.78))',\n"
           "        cardPattern: 'radial-gradient(circle at 22% 24%, rgba(129, 140, 248, 0.32) 0, transparent 46%), radial-gradient(circle at 84% 14%, rgba(165, 180, 252, 0.28) 0, transparent 40%)'\n"
           "      },\n"
           "      {\n"
           "        messageBackground: 'linear-gradient(135deg, rgba(45, 212, 191, 0.2), rgba(59, 130, 246, 0.42))',\n"
           "        border: '#5eead4',\n"
           "        glow: 'rgba(56, 189, 248, 0.38)',\n"
           "        cardBackground: 'linear-gradient(160deg, rgba(8, 47, 73, 0.85), rgba(30, 64, 175, 0.78))',\n"
           "        cardPattern: 'radial-gradient(circle at 28% 18%, rgba(56, 189, 248, 0.32) 0, transparent 42%), radial-gradient(circle at 76% 8%, rgba(45, 212, 191, 0.28) 0, transparent 38%)'\n"
           "      }\n"
           "    ];\n"
           "    const participantStyles = new Map();\n"
           "    function computePaletteEntry(index) {\n"
           "      const position = Number.isFinite(index) && index >= 0 ? index : 0;\n"
           "      if (position < basePalette.length) {\n"
           "        return { ...basePalette[position] };\n"
           "      }\n"
           "      const hue = (position * 47) % 360;\n"
           "      const accent = (hue + 60) % 360;\n"
           "      return {\n"
           "        messageBackground: `linear-gradient(135deg, hsla(${hue}, 80%, 22%, 0.55), hsla(${accent}, 85%, 36%, 0.75))`,\n"
           "        border: `hsla(${accent}, 90%, 65%, 1)`,\n"
           "        glow: `hsla(${accent}, 90%, 70%, 0.45)`,\n"
           "        cardBackground: `linear-gradient(160deg, hsla(${hue}, 65%, 16%, 0.9), hsla(${accent}, 70%, 22%, 0.95))`,\n"
           "        cardPattern: `radial-gradient(circle at 20% 25%, hsla(${accent}, 85%, 60%, 0.35) 0, transparent 42%), radial-gradient(circle at 80% 10%, hsla(${hue}, 85%, 55%, 0.3) 0, transparent 38%)`\n"
           "      };\n"
           "    }\n"
           "    function getPaletteForIndex(index) {\n"
           "      const palette = computePaletteEntry(index);\n"
           "      if (!palette.glow) {\n"
           "        palette.glow = 'rgba(59, 130, 246, 0.35)';\n"
           "      }\n"
           "      if (!palette.messageBackground) {\n"
           "        palette.messageBackground = 'linear-gradient(135deg, rgba(59, 130, 246, 0.18), rgba(129, 140, 248, 0.4))';\n"
           "      }\n"
           "      if (!palette.cardBackground) {\n"
           "        palette.cardBackground = 'linear-gradient(160deg, rgba(15, 23, 42, 0.85), rgba(30, 64, 175, 0.78))';\n"
           "      }\n"
           "      return palette;\n"
           "    }\n"
           "    function assignParticipantStyles(participants) {\n"
           "      participantStyles.clear();\n"
           "      participants.forEach((participant, index) => {\n"
           "        participantStyles.set(index, getPaletteForIndex(index));\n"
           "      });\n"
           "    }\n"
           "    function normaliseParticipants(list) {\n"
           "      if (!Array.isArray(list)) {\n"
           "        return [];\n"
           "      }\n"
           "      return list\n"
           "        .map((participant) => ({\n"
           "          name: participant && typeof participant.name === 'string' ? participant.name.trim() : '',\n"
           "          model: participant && typeof participant.model === 'string' ? participant.model.trim() : '',\n"
           "          displayModel: participant && typeof participant.displayModel === 'string' ? participant.displayModel.trim() : ''\n"
           "        }))\n"
           "        .filter((participant) => participant.name || participant.model || participant.displayModel);\n"
           "    }\n"
           "    function resetTranscriptState(topic, turns, participants) {\n"
           "      transcriptMessages.length = 0;\n"
           "      summaryAppended = false;\n"
           "      currentTopic = typeof topic === 'string' ? topic : '';\n"
           "      currentTurns = Number.isFinite(turns) ? turns : 0;\n"
           "      currentParticipants = normaliseParticipants(participants);\n"
           "    }\n"
           "    function recordTranscriptMessage(message) {\n"
           "      if (!message || message.isSummary) {\n"
           "        return;\n"
           "      }\n"
           "      const entry = {\n"
           "        name: typeof message.name === 'string' ? message.name : '',\n"
           "        text: typeof message.text === 'string' ? message.text : ''\n"
           "      };\n"
           "      transcriptMessages.push(entry);\n"
           "    }\n"
           "    function extractPlainText(html) {\n"
           "      if (!html) {\n"
           "        return '';\n"
           "      }\n"
           "      const temp = document.createElement('div');\n"
           "      temp.innerHTML = html;\n"
           "      const text = temp.textContent || temp.innerText || '';\n"
           "      return text.replace(/\\s+/g, ' ').trim();\n"
           "    }\n"
           "    function summariseDiscussionIfNeeded() {\n"
           "      if (summaryAppended || transcriptMessages.length === 0) {\n"
           "        return;\n"
           "      }\n"
           "      const participantsList = Array.isArray(currentParticipants) ? currentParticipants : [];\n"
           "      const participantNames = participantsList\n"
           "        .map((participant) => (participant && participant.name ? participant.name : ''))\n"
           "        .filter((name) => name);\n"
           "      const subject = participantNames.length ? participantNames.join(', ') : 'The participants';\n"
           "      const topicText = currentTopic ? currentTopic : 'the selected topic';\n"
           "      const turnsValue = Number.isFinite(currentTurns) && currentTurns > 0 ? currentTurns : transcriptMessages.length;\n"
           "      const turnsText = turnsValue === 1 ? '1 turn' : `${turnsValue} turns`;\n"
           "      const latestByParticipant = new Map();\n"
           "      transcriptMessages.forEach((entry) => {\n"
           "        if (entry.name) {\n"
           "          latestByParticipant.set(entry.name, entry.text || '');\n"
           "        }\n"
           "      });\n"
           "      const highlightSnippets = [];\n"
           "      latestByParticipant.forEach((text, name) => {\n"
           "        const plain = extractPlainText(text);\n"
           "        if (!plain) {\n"
           "          return;\n"
           "        }\n"
           "        const sentenceMatch = plain.match(/[^.!?]+[.!?]?/);\n"
           "        let sentence = sentenceMatch ? sentenceMatch[0].trim() : plain.trim();\n"
           "        if (sentence.length > 160) {\n"
           "          sentence = `${sentence.slice(0, 157).replace(/\\s+$/g, '')}…`;\n"
           "        }\n"
           "        if (sentence) {\n"
           "          highlightSnippets.push(`${name}: ${sentence}`);\n"
           "        }\n"
           "      });\n"
           "      const highlightsText = highlightSnippets.length ? ` Highlights — ${highlightSnippets.join(' | ')}` : '';\n"
           "      const summaryBody = `${subject} discussed \"${topicText}\" over ${turnsText}.${highlightsText}`;\n"
           "      appendMessage({\n"
           "        participantIndex: -1,\n"
           "        name: 'Summary',\n"
           "        model: '',\n"
           "        displayModel: '',\n"
           "        text: `<p>${summaryBody}</p>`,\n"
           "        isSummary: true\n"
           "      });\n"
           "      summaryAppended = true;\n"
           "    }\n"
           "    function updateParticipantCardThemes() {\n"
          "      const cards = Array.from(participantsEl.querySelectorAll('.participant'));\n"
          "      cards.forEach((card, index) => {\n"
          "        const palette = getPaletteForIndex(index);\n"
          "        card.style.setProperty('--participant-border', palette.border);\n"
          "        card.style.setProperty('--participant-pattern', palette.cardPattern || 'rgba(59, 130, 246, 0.25)');\n"
          "        card.style.setProperty('--participant-surface', palette.cardBackground);\n"
          "        card.style.setProperty('--participant-glow', palette.glow || 'rgba(59, 130, 246, 0.35)');\n"
          "        card.style.setProperty('--participant-highlight', palette.border || 'rgba(148, 163, 184, 0.8)');\n"
          "      });\n"
          "    }\n"
           "    function appendMessage(message) {\n"
           "      if (!message || typeof message !== 'object') {\n"
           "        return;\n"
           "      }\n"
           "      const item = document.createElement('div');\n"
           "      item.className = 'message message-enter';\n"
           "      if (message.isSummary) {\n"
           "        item.classList.add('message-summary');\n"
           "      }\n"
           "      const paletteIndex = Number.isFinite(message.participantIndex) && message.participantIndex >= 0\n"
           "        ? message.participantIndex\n"
           "        : 0;\n"
           "      const paletteEntry = participantStyles.get(paletteIndex);\n"
           "      const palette = paletteEntry || getPaletteForIndex(paletteIndex);\n"
           "      item.style.setProperty('--message-bg', palette.messageBackground);\n"
           "      item.style.setProperty('--message-border', palette.border);\n"
           "      item.style.setProperty('--message-glow', palette.glow || 'rgba(59, 130, 246, 0.35)');\n"
          "      const displayModel = (message.displayModel && typeof message.displayModel === 'string') ? message.displayModel : '';\n"
           "      const canonicalModel = (message.model && typeof message.model === 'string') ? message.model : '';\n"
           "      let modelLabel = displayModel;\n"
           "      if (displayModel && canonicalModel && displayModel !== canonicalModel) {\n"
           "        modelLabel = `${displayModel} (${canonicalModel})`;\n"
           "      } else if (!modelLabel && canonicalModel) {\n"
           "        modelLabel = canonicalModel;\n"
           "      }\n"
           "      const header = modelLabel\n"
           "        ? `<strong>${message.name} <span style=\"color:#94a3b8; font-weight:400;\">(${modelLabel})</span></strong>`\n"
           "        : `<strong>${message.name}</strong>`;\n"
           "      item.innerHTML = `${header}${message.text}`;\n"
          "      messagesEl.appendChild(item);\n"
          "      recordTranscriptMessage(message);\n"
          "      item.addEventListener('animationend', (event) => {\n"
          "        if (event.animationName === 'messagePulse') {\n"
          "          item.classList.remove('message-enter');\n"
          "        }\n"
          "      });\n"
          "      requestAnimationFrame(() => {\n"
          "        requestAnimationFrame(() => {\n"
          "          item.classList.add('is-visible');\n"
          "        });\n"
          "      });\n"
          "      transcriptEl.style.display = 'block';\n"
          "      transcriptEl.scrollTop = transcriptEl.scrollHeight;\n"
          "    }\n"
          "    function populateModelOptions(select, selectedModel) {\n"
          "      const datasetValue = (select.dataset.desiredModel || '').trim();\n"
          "      const datasetDisplay = (select.dataset.displayModel || '').trim();\n"
          "      const providedValue = (selectedModel && typeof selectedModel === 'string') ? selectedModel.trim() : '';\n"
          "      const currentValue = (select.value && typeof select.value === 'string') ? select.value.trim() : '';\n"
          "      const requestedValue = providedValue || datasetValue || currentValue;\n"
          "      const requestedDisplay = datasetDisplay || requestedValue;\n"
          "      const suggestedValue = (select.dataset.suggestedModel || '').trim();\n"
          "      select.innerHTML = '';\n"
          "      const placeholder = document.createElement('option');\n"
          "      placeholder.value = '';\n"
          "      placeholder.textContent = modelLoadError\n"
           "        ? 'Unable to load models'\n"
           "        : (availableModels.length ? 'Select a model' : 'Loading models...');\n"
           "      placeholder.disabled = availableModels.length > 0;\n"
           "      select.appendChild(placeholder);\n"
           "      let hasMatch = false;\n"
           "      let matchedDisplay = '';\n"
           "      availableModels.forEach((item) => {\n"
           "        const option = document.createElement('option');\n"
           "        const canonical = item && typeof item.model === 'string' ? item.model : '';\n"
           "        const alias = item && typeof item.name === 'string' && item.name ? item.name : canonical;\n"
           "        option.value = canonical;\n"
           "        option.dataset.canonicalModel = canonical;\n"
           "        option.dataset.displayModel = alias;\n"
           "        option.textContent = alias && canonical && alias !== canonical\n"
           "          ? `${alias} (${canonical})`\n"
           "          : (alias || canonical || '');\n"
           "        if ((canonical && canonical === requestedValue) || (!canonical && alias === requestedValue) || (!hasMatch && alias === requestedValue)) {\n"
           "          option.selected = true;\n"
           "          hasMatch = true;\n"
           "          matchedDisplay = alias || canonical;\n"
           "        }\n"
           "        select.appendChild(option);\n"
           "      });\n"
          "      if (availableModels.length === 0) {\n"
          "        placeholder.selected = true;\n"
          "        if (requestedValue) {\n"
          "          select.dataset.desiredModel = requestedValue;\n"
          "        } else {\n"
           "          delete select.dataset.desiredModel;\n"
           "        }\n"
           "        if (requestedDisplay) {\n"
           "          select.dataset.displayModel = requestedDisplay;\n"
           "        } else {\n"
           "          delete select.dataset.displayModel;\n"
           "        }\n"
          "      } else if (hasMatch) {\n"
          "        select.dataset.desiredModel = requestedValue;\n"
          "        select.dataset.displayModel = matchedDisplay || requestedDisplay || requestedValue;\n"
          "        if (requestedValue && suggestedValue === requestedValue) {\n"
          "          delete select.dataset.suggestedModel;\n"
          "        }\n"
          "      } else {\n"
          "        placeholder.selected = true;\n"
          "        if (requestedValue && !missingModelWarning && requestedValue !== suggestedValue) {\n"
          "          missingModelWarning = true;\n"
          "          if (!statusEl.textContent) {\n"
          "            setStatus('A previously selected model is no longer available.');\n"
          "          }\n"
           "        }\n"
           "        if (requestedValue) {\n"
           "          select.dataset.desiredModel = requestedValue;\n"
           "        } else {\n"
           "          delete select.dataset.desiredModel;\n"
           "        }\n"
           "        if (requestedDisplay) {\n"
           "          select.dataset.displayModel = requestedDisplay;\n"
           "        } else {\n"
           "          delete select.dataset.displayModel;\n"
           "        }\n"
           "      }\n"
           "    }\n"
          "    function registerModelSelect(select, selectedModel, isSuggestion) {\n"
          "      const suggestion = Boolean(isSuggestion && selectedModel);\n"
          "      if (selectedModel) {\n"
          "        select.dataset.desiredModel = selectedModel;\n"
          "        if (!select.dataset.displayModel) {\n"
          "          select.dataset.displayModel = selectedModel;\n"
          "        }\n"
          "        if (suggestion) {\n"
          "          select.dataset.suggestedModel = selectedModel;\n"
          "        } else {\n"
          "          delete select.dataset.suggestedModel;\n"
          "        }\n"
          "      } else {\n"
          "        delete select.dataset.desiredModel;\n"
          "        delete select.dataset.displayModel;\n"
          "        delete select.dataset.suggestedModel;\n"
          "      }\n"
          "      select.addEventListener('change', () => {\n"
          "        if (select.value) {\n"
          "          select.dataset.desiredModel = select.value;\n"
           "        } else {\n"
           "          delete select.dataset.desiredModel;\n"
           "        }\n"
           "        const currentOption = select.options[select.selectedIndex];\n"
           "        if (currentOption && currentOption.dataset && currentOption.dataset.displayModel) {\n"
           "          select.dataset.displayModel = currentOption.dataset.displayModel;\n"
           "        } else if (select.value) {\n"
           "          select.dataset.displayModel = select.value;\n"
          "        } else {\n"
          "          delete select.dataset.displayModel;\n"
          "        }\n"
          "        delete select.dataset.suggestedModel;\n"
          "        if (statusEl.textContent === 'A previously selected model is no longer available.') {\n"
          "          setStatus('');\n"
          "        }\n"
          "      });\n"
          "      modelSelects.add(select);\n"
          "      populateModelOptions(select, selectedModel);\n"
          "    }\n"
           "    function unregisterModelSelect(select) {\n"
            "      modelSelects.delete(select);\n"
            "    }\n"
            "    function refreshModelSelects() {\n"
            "      modelSelects.forEach((select) => {\n"
           "        const desired = select.dataset.desiredModel || select.value;\n"
           "        populateModelOptions(select, desired);\n"
            "      });\n"
            "    }\n"
           "    async function loadModels() {\n"
           "      missingModelWarning = false;\n"
          "      if (statusEl.textContent === 'A previously selected model is no longer available.') {\n"
          "        setStatus('');\n"
          "      }\n"
           "      try {\n"
           "        const response = await fetch('/models');\n"
           "        if (!response.ok) {\n"
           "          throw new Error('Request failed');\n"
           "        }\n"
           "        const payload = await response.json();\n"
           "        availableModels = Array.isArray(payload.models) ? payload.models : [];\n"
           "        modelLoadError = false;\n"
          "        if (availableModels.length && statusEl.textContent === 'Unable to load models from Ollama.') {\n"
          "          setStatus('');\n"
          "        }\n"
           "      } catch (error) {\n"
           "        availableModels = [];\n"
           "        modelLoadError = true;\n"
          "        if (!statusEl.textContent) {\n"
          "          setStatus('Unable to load models from Ollama.');\n"
          "        }\n"
           "      }\n"
           "      refreshModelSelects();\n"
           "    }\n"
          "    function createParticipant(name, model, isSuggestion) {\n"
          "      const wrapper = document.createElement('div');\n"
          "      wrapper.className = 'participant';\n"
          "      wrapper.innerHTML = `\n"
          "        <label>Friendly name</label>\n"
          "        <input name=\"name\" placeholder=\"Astra\" value=\"${name || ''}\" />\n"
          "        <label>Ollama model</label>\n"
          "        <select name=\"model\"></select>\n"
          "        <button type=\"button\" class=\"remove\">Remove</button>\n"
          "      `;\n"
          "      const select = wrapper.querySelector('select[name=\"model\"]');\n"
          "      registerModelSelect(select, model || '', isSuggestion);\n"
          "      wrapper.querySelector('.remove').addEventListener('click', () => {\n"
          "        unregisterModelSelect(select);\n"
          "        participantsEl.removeChild(wrapper);\n"
          "        updateParticipantCardThemes();\n"
          "      });\n"
           "      participantsEl.appendChild(wrapper);\n"
           "      updateParticipantCardThemes();\n"
           "    }\n"
          "    document.getElementById('addParticipant').addEventListener('click', (event) => {\n"
          "      event.preventDefault();\n"
          "      createParticipant('', '', false);\n"
          "    });\n"
           "    document.getElementById('start').addEventListener('click', async (event) => {\n"
           "      event.preventDefault();\n"
          "      setStatus('');\n"
           "      messagesEl.innerHTML = '';\n"
           "      participantStyles.clear();\n"
           "      transcriptEl.style.display = 'none';\n"
           "      const topic = document.getElementById('topic').value.trim();\n"
           "      const turns = parseInt(document.getElementById('turns').value, 10);\n"
           "      const participantDivs = participantsEl.querySelectorAll('.participant');\n"
           "      const participants = [];\n"
           "      participantDivs.forEach((div, index) => {\n"
           "        const name = div.querySelector('input[name=\"name\"]').value.trim();\n"
           "        const selectEl = div.querySelector('select[name=\"model\"]');\n"
           "        const rawValue = (selectEl && typeof selectEl.value === 'string') ? selectEl.value.trim() : '';\n"
           "        const selectedOption = selectEl ? selectEl.options[selectEl.selectedIndex] : null;\n"
           "        const optionDisplay = (selectedOption && selectedOption.dataset && typeof selectedOption.dataset.displayModel === 'string')\n"
           "          ? selectedOption.dataset.displayModel.trim()\n"
           "          : '';\n"
           "        const optionCanonical = (selectedOption && selectedOption.dataset && typeof selectedOption.dataset.canonicalModel === 'string')\n"
           "          ? selectedOption.dataset.canonicalModel.trim()\n"
           "          : '';\n"
           "        const datasetDisplay = (selectEl && selectEl.dataset && typeof selectEl.dataset.displayModel === 'string')\n"
           "          ? selectEl.dataset.displayModel.trim()\n"
           "          : '';\n"
           "        const canonicalValue = optionCanonical || rawValue;\n"
           "        const displayValue = optionDisplay || datasetDisplay || canonicalValue;\n"
           "        if (canonicalValue) {\n"
           "          participants.push({\n"
           "            name: name || `Companion ${index + 1}`,\n"
           "            model: canonicalValue,\n"
           "            displayModel: displayValue\n"
           "          });\n"
           "        }\n"
           "      });\n"
          "      if (!topic) {\n"
          "        setStatus('Please provide a topic.');\n"
          "        return;\n"
          "      }\n"
          "      if (Number.isNaN(turns) || turns < 1) {\n"
          "        setStatus('Please provide a valid number of turns.');\n"
          "        return;\n"
          "      }\n"
          "      if (participants.length === 0) {\n"
          "        setStatus('Add at least one participant with a model selected.');\n"
          "        return;\n"
          "      }\n"
          "      resetTranscriptState(topic, turns, participants);\n"
          "      setStatus('Starting conversation...');\n"
           "      try {\n"
           "        const response = await fetch('/chat', {\n"
           "          method: 'POST',\n"
           "          headers: { 'Content-Type': 'application/json' },\n"
           "          body: JSON.stringify({ topic, turns, participants })\n"
           "        });\n"
          "        if (!response.ok) {\n"
          "          let payload = null;\n"
          "          try {\n"
          "            payload = await response.json();\n"
          "          } catch (parseError) {\n"
          "            // ignore JSON parse errors\n"
          "          }\n"
          "          const errorText = payload && typeof payload.error === 'string' && payload.error\n"
          "            ? payload.error\n"
          "            : 'The conversation failed.';\n"
          "          setStatus(errorText);\n"
          "          return;\n"
          "        }\n"
          "        const reader = response.body && response.body.getReader ? response.body.getReader() : null;\n"
          "        if (!reader) {\n"
          "          setStatus('Streaming is not supported by this browser.');\n"
          "          return;\n"
          "        }\n"
          "        setStatus('Waiting for responses...');\n"
           "        transcriptEl.style.display = 'block';\n"
           "        const decoder = new TextDecoder();\n"
           "        let buffer = '';\n"
           "        let stopStreaming = false;\n"
           "        let encounteredError = false;\n"
           "        while (!stopStreaming) {\n"
           "          const { value, done } = await reader.read();\n"
           "          if (done) {\n"
           "            break;\n"
           "          }\n"
           "          buffer += decoder.decode(value, { stream: true });\n"
           "          const lines = buffer.split('\\n');\n"
           "          buffer = lines.pop();\n"
           "          for (const line of lines) {\n"
           "            const trimmed = line.trim();\n"
           "            if (!trimmed) {\n"
           "              continue;\n"
           "            }\n"
           "            let eventPayload;\n"
           "            try {\n"
           "              eventPayload = JSON.parse(trimmed);\n"
           "            } catch (parseError) {\n"
           "              continue;\n"
           "            }\n"
          "            if (eventPayload.type === 'start') {\n"
          "              const participantsList = Array.isArray(eventPayload.participants) ? eventPayload.participants : [];\n"
          "              assignParticipantStyles(participantsList);\n"
          "              currentParticipants = normaliseParticipants(participantsList);\n"
          "              if (typeof eventPayload.topic === 'string') {\n"
          "                currentTopic = eventPayload.topic;\n"
          "              }\n"
          "              if (Number.isFinite(eventPayload.turns)) {\n"
          "                currentTurns = eventPayload.turns;\n"
          "              }\n"
          "              setStatus('Conversation in progress...');\n"
          "            } else if (eventPayload.type === 'message' && eventPayload.message) {\n"
          "              appendMessage(eventPayload.message);\n"
          "              setStatus(`Responding: ${eventPayload.message.name}`);\n"
          "            } else if (eventPayload.type === 'error') {\n"
          "              const errorMessage = eventPayload.message && typeof eventPayload.message === 'string' && eventPayload.message\n"
          "                ? eventPayload.message\n"
          "                : 'The conversation failed.';\n"
          "              encounteredError = true;\n"
          "              setStatus(errorMessage);\n"
          "              stopStreaming = true;\n"
          "              break;\n"
          "            } else if (eventPayload.type === 'complete') {\n"
          "              if (typeof eventPayload.topic === 'string') {\n"
          "                currentTopic = eventPayload.topic;\n"
          "              }\n"
          "              if (Number.isFinite(eventPayload.turns)) {\n"
          "                currentTurns = eventPayload.turns;\n"
          "              }\n"
          "              summariseDiscussionIfNeeded();\n"
          "              setStatus('Conversation complete.');\n"
          "              stopStreaming = true;\n"
          "              break;\n"
          "            }\n"
           "          }\n"
           "          if (stopStreaming) {\n"
           "            await reader.cancel().catch(() => {});\n"
           "            break;\n"
           "          }\n"
           "        }\n"
           "        if (!stopStreaming) {\n"
           "          buffer += decoder.decode();\n"
           "          const trimmed = buffer.trim();\n"
          "          if (trimmed) {\n"
          "            try {\n"
          "              const eventPayload = JSON.parse(trimmed);\n"
          "              if (eventPayload.type === 'error') {\n"
          "                const errorMessage = eventPayload.message && typeof eventPayload.message === 'string' && eventPayload.message\n"
          "                  ? eventPayload.message\n"
          "                  : 'The conversation failed.';\n"
          "                encounteredError = true;\n"
          "                setStatus(errorMessage);\n"
          "              } else if (eventPayload.type === 'complete') {\n"
          "                if (typeof eventPayload.topic === 'string') {\n"
          "                  currentTopic = eventPayload.topic;\n"
          "                }\n"
          "                if (Number.isFinite(eventPayload.turns)) {\n"
          "                  currentTurns = eventPayload.turns;\n"
          "                }\n"
          "                summariseDiscussionIfNeeded();\n"
          "                setStatus('Conversation complete.');\n"
          "              }\n"
          "            } catch (parseError) {\n"
           "              // ignore trailing parse issues\n"
           "            }\n"
           "          }\n"
           "        }\n"
           "        if (!encounteredError) {\n"
           "          summariseDiscussionIfNeeded();\n"
           "        }\n"
          "      } catch (error) {\n"
          "        setStatus('Unable to reach the aiChat server.');\n"
          "      }\n"
           "    });\n"
          "    createParticipant('Astra', 'gemma:2b', true);\n"
          "    createParticipant('Nova', 'llama3:8b', true);\n"
          "    loadModels();\n"
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

    memset(participants, 0, sizeof(participants));
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
        json_object *display_obj = NULL;
        const char *name = NULL;
        const char *model = NULL;
        const char *display = NULL;

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

        if (json_object_object_get_ex(item, "displayModel", &display_obj) &&
            json_object_get_type(display_obj) == json_type_string && json_object_get_string_len(display_obj) > 0) {
            display = json_object_get_string(display_obj);
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
        if (display && *display) {
            strncpy(participants[participant_count].display_model, display, MAX_MODEL_LENGTH - 1);
            participants[participant_count].display_model[MAX_MODEL_LENGTH - 1] = '\0';
        }
        participant_count++;
    }

    json_object_put(payload);

    int needs_lookup = 0;
    for (size_t i = 0; i < participant_count; ++i) {
        if (participants[i].display_model[0] == '\0') {
            needs_lookup = 1;
            break;
        }
    }

    json_object *models_payload = NULL;
    if (needs_lookup && fetch_available_models(ollama_url, &models_payload, NULL) != 0) {
        models_payload = NULL;
    }
    ensure_participant_display_models(participants, participant_count, models_payload);
    if (models_payload) {
        json_object_put(models_payload);
    }

    if (participant_count == 0) {
        send_http_error(client_fd, "400 Bad Request", "No valid participants supplied.");
        return;
    }

    if (send_chunked_header(client_fd, "200 OK", "application/x-ndjson") != 0) {
        return;
    }

    json_object *start_event = json_object_new_object();
    json_object *start_participants = json_object_new_array();
    if (!start_event || !start_participants) {
        if (start_event) {
            json_object_put(start_event);
        }
        if (start_participants) {
            json_object_put(start_participants);
        }
        send_stream_error_event(client_fd, "Failed to start stream.");
        finish_chunked_response(client_fd);
        return;
    }

    for (size_t i = 0; i < participant_count; ++i) {
        json_object *participant_obj = json_object_new_object();
        if (!participant_obj) {
            json_object_put(start_participants);
            json_object_put(start_event);
            send_stream_error_event(client_fd, "Failed to start stream.");
            finish_chunked_response(client_fd);
            return;
        }
        json_object_object_add(participant_obj, "name", json_object_new_string(participants[i].name));
        json_object_object_add(participant_obj, "model", json_object_new_string(participants[i].model));
        if (participants[i].display_model[0] != '\0') {
            json_object_object_add(participant_obj, "displayModel",
                                   json_object_new_string(participants[i].display_model));
        }
        json_object_array_add(start_participants, participant_obj);
    }

    json_object_object_add(start_event, "type", json_object_new_string("start"));
    json_object_object_add(start_event, "topic", json_object_new_string(topic));
    json_object_object_add(start_event, "turns", json_object_new_int(turns));
    json_object_object_add(start_event, "participants", start_participants);

    if (send_json_chunk(client_fd, start_event) != 0) {
        json_object_put(start_event);
        finish_chunked_response(client_fd);
        return;
    }
    json_object_put(start_event);

    struct StreamContext stream_ctx = {client_fd, 0};
    if (run_conversation(topic, turns, participants, participant_count, ollama_url, stream_message_callback,
                         &stream_ctx, &result, &error_message) != 0) {
        if (!stream_ctx.failed) {
            if (error_message) {
                send_stream_error_event(client_fd, error_message);
            } else {
                send_stream_error_event(client_fd, "Conversation failed.");
            }
            finish_chunked_response(client_fd);
        }
        if (error_message) {
            free(error_message);
        }
        if (result) {
            json_object_put(result);
        }
        return;
    }

    if (!stream_ctx.failed) {
        json_object *complete_event = json_object_new_object();
        if (complete_event) {
            json_object_object_add(complete_event, "type", json_object_new_string("complete"));
            json_object_object_add(complete_event, "topic", json_object_new_string(topic));
            json_object_object_add(complete_event, "turns", json_object_new_int(turns));
            if (send_json_chunk(client_fd, complete_event) != 0) {
                stream_ctx.failed = 1;
            }
            json_object_put(complete_event);
        }
    }

    if (!stream_ctx.failed) {
        finish_chunked_response(client_fd);
    }

    if (result) {
        json_object_put(result);
    }
    if (error_message) {
        free(error_message);
    }
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
    } else if (strcmp(method, "GET") == 0 && strcmp(path, "/models") == 0) {
        handle_models_request(client_fd, ollama_url);
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

    for (int attempt = 0; attempt <= FALLBACK_PORT_STEPS; ++attempt) {
        address.sin_port = htons((uint16_t)port);
        if (bind(server_fd, (struct sockaddr *)&address, sizeof(address)) == 0) {
            if (attempt > 0) {
                fallback_used = 1;
            }
            break;
        }

        if (!(errno == EADDRINUSE && !port_from_env && attempt < FALLBACK_PORT_STEPS)) {
            perror("bind");
            close(server_fd);
            return EXIT_FAILURE;
        }

        int next_port = DEFAULT_PORT + attempt + 1;
        fprintf(stderr, "Port %d unavailable, trying %d instead.\n", port, next_port);
        port = next_port;
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
        printf("Port %d unavailable, using fallback port %d.\n", requested_port, port);
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
