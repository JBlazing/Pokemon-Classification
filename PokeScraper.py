'''
	This python script will scrape the pokemon type and image
	and store the type into a csv and save the artwork in the directory

'''


import requests


file = open('pokemans.csv' , 'w')

file.close()


for i in range(1,802):
	url = "http://pokeapi.co/api/v2/pokemon/{}/".format(i)
	r = requests.get(url);
	if r.status_code == 200:
		j = r.json()
		name = j["name"];
		
		types = j['types']
		miniL = []
		for type in types:
			miniL.append(type['type']['name'])
		s = ','.join(miniL)
		file.write(s)
		file.write('\n')
		
		link = "https://img.pokemondb.net/artwork/vector/{}.png".format(name)
		rr = requests.get(link , stream=True)
		f_name = "VPokemon/A%04d.png" % (i)
		with open(f_name, 'wb') as f: 
			for chunk in rr.iter_content(chunk_size = 1024*1024): 
				if chunk: 
					f.write(chunk)
		