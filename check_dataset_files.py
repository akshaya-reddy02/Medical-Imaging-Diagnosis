from kaggle.api.kaggle_api_extended import KaggleApi

DATASET = "mohammadzunaed/nih-chest-x-ray14"

api = KaggleApi()
api.authenticate()

token = None
page = 0
total = 0
found = []

while True:
    response = api.dataset_list_files(
        DATASET,
        page_token=token,
        page_size=200,
    )

    files = response._dataset_files
    page += 1
    total += len(files)

    for file in files:
        name = getattr(file, "name", "")

        if name.lower().endswith(
            (".csv", ".txt", ".json", ".md", ".xlsx")
        ):
            found.append(name)

    print(f"Page {page}: checked {total} files")

    token = response._next_page_token

    if not token:
        break

print("\nMetadata and label files:")
if found:
    for name in found:
        print(name)
else:
    print("No matching files found.")

print(f"\nTotal files checked: {total}")