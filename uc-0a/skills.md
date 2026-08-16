skills:
  - name: classify_complaint
    description: "Classifies a single complaint into category, priority, reason, and flag using LLM and strict enforcement rules."
    input: "A dictionary representing one complaint row containing 'description' and 'location'."
    output: "A dictionary with keys 'category', 'priority', 'reason', and 'flag'."
    error_handling: "If the description is vague or short, classify category as 'Other' and set flag to 'NEEDS_REVIEW'."

  - name: batch_classify
    description: "Processes a CSV file of complaints, classifies each row, and writes the results to an output CSV file."
    input: "Paths to the input CSV file and output CSV file."
    output: "Writes a CSV file with classified complaints."
    error_handling: "Log and skip malformed rows, continuing classification for the remaining rows."
