package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"os"

	"github.com/ollama/ollama/api"
)

type Character struct {
	Name string `json:"name"`
	Kind string `json:"kind"`
}

func main() {

	ctx := context.Background()

	ollamaUrl := os.Getenv("OLLAMA_HOST")
	model := os.Getenv("LLM")

	fmt.Println("🌍", ollamaUrl, "📕", model)

	client, err := api.ClientFromEnvironment()
	if err != nil {
		log.Fatal("😡:", err)
	}

	systemInstructions := `You are an expert NPC generator for games like D&D. 
	You have freedom to be creative to get the best possible output.
	`

	// define schema for a structured output
	// ref: https://ollama.com/blog/structured-outputs
	schema := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"name": map[string]any{
				"type": "string",
			},
			"kind": map[string]any{
				"type": "string",
			},
		},
		"required": []string{"name", "kind"},
	}

	jsonModel, err := json.Marshal(schema)
	if err != nil {
		log.Fatalln("😡", err)
	}

	kind := "Dwarf"
	//kind := "Human"
	//kind := "Elf"
	userContent := fmt.Sprintf("Generate a random name for an %s (kind always equals %s).", kind, kind)

	// Prompt construction
	messages := []api.Message{
		{Role: "system", Content: systemInstructions},
		{Role: "user", Content: userContent},
	}

	//stream := true
	noStream := false

	req := &api.ChatRequest{
		Model:    model,
		Messages: messages,
		Options: map[string]interface{}{
			"temperature":    1.7,
			"repeat_last_n":  2,
			"repeat_penalty": 2.2,
			"top_k":          10,
			"top_p":          0.9,
		},
		Format: json.RawMessage(jsonModel),
		Stream: &noStream,
	}

	generateName := func() (string, error) {
		jsonResult := ""
		respFunc := func(resp api.ChatResponse) error {
			jsonResult = resp.Message.Content
			return nil
		}
		// Start the chat completion
		err := client.Chat(ctx, req, respFunc)
		if err != nil {
			return jsonResult, err
		}
		return jsonResult, nil
	}

	characters := []Character{}
	for i := 0; i < 15; i++ {
		// Generate a random name
		jsonStr, err := generateName()
		if err != nil {
			log.Fatal("😡:", err)
		}
		character := Character{}

		err = json.Unmarshal([]byte(jsonStr), &character)
		if err != nil {
			log.Fatal("😡:", err)
		}
		fmt.Println(character.Name, character.Kind)

		characters = append(characters, character)
	}

	// Create a Markdown table
	markdownTable := "| Index | Name     | Kind       |\n"
	markdownTable += "|------|----------|------------|\n"

	// Add rows to the Markdown table
	for idx, character := range characters {
		markdownTable += fmt.Sprintf("| %d   | %s      | %s       |\n", idx+1, character.Name, character.Kind)
	}

	// Write the Markdown table to a file
	err = os.WriteFile("./characters."+kind+".md", []byte(markdownTable), 0644)
	if err != nil {
		log.Fatal("😡:", err)
	}

}
