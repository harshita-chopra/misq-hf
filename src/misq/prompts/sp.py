# sp.py in prompts
# This module provides prompt templates or prompt-building utilities for the sp task in MISQ.

def get_sp_prompt(item):
    """
    Generate a prompt string for a single item from sp.json.
    Args:
        item (dict): The item for which to generate the prompt.
    Returns:
        str: The formatted prompt string.
    """
    # Example: customize this template as needed for your MISQ use case
    prompt = f"Process the following item: {item}"
    return prompt

# Example usage (for testing):
if __name__ == "__main__":
    example_item = {'id': 1, 'text': 'Sample item'}
    print(get_sp_prompt(example_item))
