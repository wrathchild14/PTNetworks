import os

folder_path = ""  # Replace with the path to your folder

for filename in os.listdir(folder_path):
    base_name, extension = os.path.splitext(filename)
    name_parts = base_name.split("_")

    if len(name_parts) > 1:
        last_numeric_index = None
        for i, part in enumerate(name_parts[::-1]):
            if part.isdigit():
                last_numeric_index = len(name_parts) - i - 1
                break

        if last_numeric_index is not None:
            counter = int(name_parts[last_numeric_index])

            # increase by 1000
            counter += 1000

            name_parts[last_numeric_index] = str(counter)
            new_name = f"{'_'.join(name_parts)}{extension}"
            os.rename(os.path.join(folder_path, filename), os.path.join(folder_path, new_name))
