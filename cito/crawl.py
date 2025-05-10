import requests
from bs4 import BeautifulSoup
import csv

def crawl_webpage(url):
    # Send HTTP request to the webpage
    response = requests.get(url)

    # Check if the request was successful
    if response.status_code != 200:
        print(f"Failed to retrieve the webpage. Status code: {response.status_code}")
        return

    # Parse the HTML content of the page using BeautifulSoup
    soup = BeautifulSoup(response.content, 'html.parser')

    # Find the parent div with the id 'objectproperties'
    parent_div = soup.find('div', id='objectproperties')

    if not parent_div:
        print("The parent div with the specified ID was not found on the page.")
        return

    # Find all the divs with the class 'entity' inside the parent div
    entity_divs = parent_div.find_all('div', class_='entity')

    if not entity_divs:
        print("No entity divs found inside the parent div.")
        return

    # Prepare the CSV file for writing
    with open('extracted_data.csv', 'w', newline='', sep='\t', encoding='utf-8') as csvfile:
        fieldnames = ['Title', 'IRI', 'Description', 'Example']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # Write header to the CSV file
        writer.writeheader()

        # Loop through each entity div and extract the relevant data
        for entity_div in entity_divs:
            # Extract title
            title = entity_div.find('h3').contents[0].strip() if entity_div.find('h3') else None
            
            # Extract IRI
            iri = entity_div.find('strong', string='IRI:').find_next('p').get_text(strip=True).replace('IRI:', '').strip() if entity_div.find('strong', string='IRI:') else None

            # Extract description
            description = entity_div.find('div', class_='comment').get_text(strip=True) if entity_div.find('div', class_='comment') else None

            # Extract example
            info_div = entity_div.find('div', class_='info')
            example = info_div.get_text(strip=True) if info_div else None

            # Write the extracted data to the CSV file
            writer.writerow({
                'Title': title,
                'IRI': iri,
                'Description': description,
                'Example': example
            })

    print("Data has been successfully written to 'extracted_data.csv'.")



# Example usage:
url = 'https://sparontologies.github.io/cito/current/cito.html'  # Replace with the actual URL
crawl_webpage(url)
