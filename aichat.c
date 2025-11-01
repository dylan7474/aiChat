#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <curl/curl.h>
#include <json-c/json.h>

// --- Configuration ---
#define OLLAMA_URL "http://192.168.50.5:11434/api/generate"
#define MODEL_1 "gemma:2b"
#define MODEL_2 "gpt-oss:latest"
#define CONVERSATION_TURNS 5 // The number of times EACH model will speak

// A simpler, more effective system prompt
#define SYSTEM_PROMPT "You are a helpful and creative AI assistant in a conversation with another AI. The user has started the conversation with a topic. Engage in a natural, back-and-forth discussion, building on what the other AI says. Keep your responses concise.\n\n"

// Struct to hold the response data from curl
struct MemoryStruct {
    char *memory;
    size_t size;
};

// Callback function for curl to write data into our struct
static size_t
WriteMemoryCallback(void *contents, size_t size, size_t nmemb, void *userp) {
    size_t realsize = size * nmemb;
    struct MemoryStruct *mem = (struct MemoryStruct *)userp;
    char *ptr = realloc(mem->memory, mem->size + realsize + 1);
    if (!ptr) {
        printf("Error: not enough memory (realloc returned NULL)\n");
        return 0;
    }
    mem->memory = ptr;
    memcpy(&(mem->memory[mem->size]), contents, realsize);
    mem->size += realsize;
    mem->memory[mem->size] = 0;
    return realsize;
}

// Function to parse the AI's JSON response and extract the text
char* parse_ollama_response(const char *json_string) {
    struct json_object *parsed_json, *response_obj, *error_obj;
    char *response_text = NULL;

    parsed_json = json_tokener_parse(json_string);
    if (!parsed_json) {
        fprintf(stderr, "Error: Could not parse JSON response.\n");
        return NULL;
    }

    if (json_object_object_get_ex(parsed_json, "error", &error_obj)) {
        const char *error_msg = json_object_get_string(error_obj);
        if (error_msg) {
            fprintf(stderr, "\nError from AI server: %s\n", error_msg);
        }
    }
    else if (json_object_object_get_ex(parsed_json, "response", &response_obj)) {
        const char *response_str = json_object_get_string(response_obj);
        if (response_str) {
            response_text = strdup(response_str);
        }
    }

    json_object_put(parsed_json);
    return response_text;
}

// Main function to get a response from a specific AI model
char* get_ai_response(const char* full_prompt, const char* model_name) {
    CURL *curl;
    char *response = NULL;
    struct MemoryStruct chunk = { .memory = malloc(1), .size = 0 };

    curl_global_init(CURL_GLOBAL_ALL);
    curl = curl_easy_init();
    if (curl) {
        json_object *jobj = json_object_new_object();
        json_object_object_add(jobj, "model", json_object_new_string(model_name));
        json_object_object_add(jobj, "prompt", json_object_new_string(full_prompt));
        json_object_object_add(jobj, "stream", json_object_new_boolean(0));

        const char *json_payload = json_object_to_json_string(jobj);
        struct curl_slist *headers = curl_slist_append(NULL, "Content-Type: application/json");

        curl_easy_setopt(curl, CURLOPT_URL, OLLAMA_URL);
        curl_easy_setopt(curl, CURLOPT_POSTFIELDS, json_payload);
        curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteMemoryCallback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, (void *)&chunk);

        printf("\033[3;90m(%s is thinking...)\033[0m", model_name);
        fflush(stdout);

        CURLcode res = curl_easy_perform(curl);
        printf("\r%*s\r", 40, ""); // Clear the "thinking" line

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

// Helper function to safely append to the history
char* append_to_history(char* history, const char* text) {
    size_t old_len = history ? strlen(history) : 0;
    size_t text_len = strlen(text);
    char* new_history = realloc(history, old_len + text_len + 1);
    if (!new_history) {
        fprintf(stderr, "Failed to reallocate memory for history\n");
        return history;
    }
    strcpy(new_history + old_len, text);
    return new_history;
}


int main() {
    char user_topic[2048];
    char* conversation_history = NULL;

    printf("--- AI Chat Arena ---\n");
    printf("Model 1 (Gemma): \033[1;34m%s\033[0m\n", MODEL_1);
    printf("Model 2 (GPT-OSS): \033[1;32m%s\033[0m\n\n", MODEL_2);
    printf("Enter the initial talking point: ");

    if (fgets(user_topic, sizeof(user_topic), stdin) == NULL) return 1;
    user_topic[strcspn(user_topic, "\n")] = 0;

    conversation_history = strdup(SYSTEM_PROMPT);
    conversation_history = append_to_history(conversation_history, "USER: ");
    conversation_history = append_to_history(conversation_history, user_topic);

    printf("\n--- Conversation Begins ---\n");

    for (int i = 0; i < CONVERSATION_TURNS; i++) {
        // Model 1's turn
        conversation_history = append_to_history(conversation_history, "\n\nMODEL 1:");
        char* response1 = get_ai_response(conversation_history, MODEL_1);
        if (response1) {
            // *** UPDATED VISUALS FOR MODEL 1 ***
            printf("----------------------------------------\n");
            printf("🤖 \033[1;34m%s\033[0m says:\n", MODEL_1);
            printf("%s\n", response1);
            
            conversation_history = append_to_history(conversation_history, response1);
            free(response1);
        } else {
            printf("\033[1;31mModel 1 failed to respond. Aborting.\033[0m\n");
            break;
        }

        // Model 2's turn
        conversation_history = append_to_history(conversation_history, "\n\nMODEL 2:");
        char* response2 = get_ai_response(conversation_history, MODEL_2);
        if (response2) {
            // *** UPDATED VISUALS FOR MODEL 2 ***
            printf("----------------------------------------\n");
            printf("🧠 \033[1;32m%s\033[0m says:\n", MODEL_2); // Corrected color code
            printf("%s\n", response2);

            conversation_history = append_to_history(conversation_history, response2);
            free(response2);
        } else {
            printf("\033[1;31mModel 2 failed to respond. Aborting.\033[0m\n");
            break;
        }
    }

    printf("--- Conversation Finished ---\n");
    free(conversation_history);
    return 0;
}
