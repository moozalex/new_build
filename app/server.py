import aiohttp
import asyncio
import uvicorn
from fastai import *
from fastai.vision import *
from io import BytesIO
from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import HTMLResponse, JSONResponse
from starlette.staticfiles import StaticFiles

export_file_url = 'https://www.dropbox.com/s/6bgq8t6yextloqp/export.pkl?raw=1'
export_file_name = 'export.pkl'

classes = ['Acadian_Flycatcher',
 'American_Crow',
 'American_Goldfinch',
 'American_Pipit',
 'American_Redstart',
 'American_Three_Toed_Woodpecker',
 'Anna_Hummingbird',
 'Artic_Tern',
 'Baird_Sparrow',
 'Baltimore_Oriole',
 'Bank_Swallow',
 'Barn_Swallow',
 'Bay_Breasted_Warbler',
 'Belted_Kingfisher',
 'Bewick_Wren',
 'Black_And_White_Warbler',
 'Black_Billed_Cuckoo',
 'Black_Capped_Vireo',
 'Black_Footed_Albatross',
 'Black_Tern',
 'Black_Throated_Blue_Warbler',
 'Black_Throated_Sparrow',
 'Blue_Grosbeak',
 'Blue_Headed_Vireo',
 'Blue_Jay',
 'Blue_Winged_Warbler',
 'Boat_Tailed_Grackle',
 'Bobolink',
 'Bohemian_Waxwing',
 'Brandt_Cormorant',
 'Brewer_Blackbird',
 'Brewer_Sparrow',
 'Bronzed_Cowbird',
 'Brown_Creeper',
 'Brown_Pelican',
 'Brown_Thrasher',
 'Cactus_Wren',
 'California_Gull',
 'Canada_Warbler',
 'Cape_Glossy_Starling',
 'Cape_May_Warbler',
 'Cardinal',
 'Carolina_Wren',
 'Caspian_Tern',
 'Cedar_Waxwing',
 'Cerulean_Warbler',
 'Chestnut_Sided_Warbler',
 'Chipping_Sparrow',
 'Chuck_Will_Widow',
 'Clark_Nutcracker',
 'Clay_Colored_Sparrow',
 'Cliff_Swallow',
 'Common_Raven',
 'Common_Tern',
 'Common_Yellowthroat',
 'Crested_Auklet',
 'Dark_Eyed_Junco',
 'Downy_Woodpecker',
 'Eared_Grebe',
 'Eastern_Towhee',
 'Elegant_Tern',
 'European_Goldfinch',
 'Evening_Grosbeak',
 'Field_Sparrow',
 'Fish_Crow',
 'Florida_Jay',
 'Forsters_Tern',
 'Fox_Sparrow',
 'Frigatebird',
 'Gadwall',
 'Geococcyx',
 'Glaucous_Winged_Gull',
 'Golden_Winged_Warbler',
 'Grasshopper_Sparrow',
 'Gray_Catbird',
 'Gray_Crowned_Rosy_Finch',
 'Gray_Kingbird',
 'Great_Crested_Flycatcher',
 'Great_Grey_Shrike',
 'Green_Jay',
 'Green_Kingfisher',
 'Green_Tailed_Towhee',
 'Green_Violetear',
 'Groove_Billed_Ani',
 'Harris_Sparrow',
 'Heermann_Gull',
 'Henslow_Sparrow',
 'Herring_Gull',
 'Hooded_Merganser',
 'Hooded_Oriole',
 'Hooded_Warbler',
 'Horned_Grebe',
 'Horned_Lark',
 'Horned_Puffin',
 'House_Sparrow',
 'House_Wren',
 'Indigo_Bunting',
 'Ivory_Gull',
 'Kentucky_Warbler',
 'Laysan_Albatross',
 'Lazuli_Bunting',
 'Le_Conte_Sparrow',
 'Least_Auklet',
 'Least_Flycatcher',
 'Least_Tern',
 'Lincoln_Sparrow',
 'Loggerhead_Shrike',
 'Long_Tailed_Jaeger',
 'Louisiana_Waterthrush',
 'Magnolia_Warbler',
 'Mallard',
 'Mangrove_Cuckoo',
 'Marsh_Wren',
 'Mockingbird',
 'Mourning_Warbler',
 'Myrtle_Warbler',
 'Nashville_Warbler',
 'Nelson_Sharp_Tailed_Sparrow',
 'Nighthawk',
 'Northern_Flicker',
 'Northern_Fulmar',
 'Northern_Waterthrush',
 'Olive_Sided_Flycatcher',
 'Orange_Crowned_Warbler',
 'Orchard_Oriole',
 'Ovenbird',
 'Pacific_Loon',
 'Painted_Bunting',
 'Palm_Warbler',
 'Parakeet_Auklet',
 'Pelagic_Cormorant',
 'Philadelphia_Vireo',
 'Pied_Billed_Grebe',
 'Pied_Kingfisher',
 'Pigeon_Guillemot',
 'Pileated_Woodpecker',
 'Pine_Grosbeak',
 'Pine_Warbler',
 'Pomarine_Jaeger',
 'Prairie_Warbler',
 'Prothonotary_Warbler',
 'Purple_Finch',
 'Red_Bellied_Woodpecker',
 'Red_Breasted_Merganser',
 'Red_Cockaded_Woodpecker',
 'Red_Eyed_Vireo',
 'Red_Faced_Cormorant',
 'Red_Headed_Woodpecker',
 'Red_Legged_Kittiwake',
 'Red_Winged_Blackbird',
 'Rhinoceros_Auklet',
 'Ring_Billed_Gull',
 'Ringed_Kingfisher',
 'Rock_Wren',
 'Rose_Breasted_Grosbeak',
 'Ruby_Throated_Hummingbird',
 'Rufous_Hummingbird',
 'Rusty_Blackbird',
 'Sage_Thrasher',
 'Savannah_Sparrow',
 'Sayornis',
 'Scarlet_Tanager',
 'Scissor_Tailed_Flycatcher',
 'Scott_Oriole',
 'Seaside_Sparrow',
 'Shiny_Cowbird',
 'Slaty_Backed_Gull',
 'Song_Sparrow',
 'Sooty_Albatross',
 'Spotted_Catbird',
 'Summer_Tanager',
 'Swainson_Warbler',
 'Tennessee_Warbler',
 'Tree_Sparrow',
 'Tree_Swallow',
 'Tropical_Kingbird',
 'Vermilion_Flycatcher',
 'Vesper_Sparrow',
 'Warbling_Vireo',
 'Western_Grebe',
 'Western_Gull',
 'Western_Meadowlark',
 'Western_Wood_Pewee',
 'Whip_Poor_Will',
 'White_Breasted_Kingfisher',
 'White_Breasted_Nuthatch',
 'White_Crowned_Sparrow',
 'White_Eyed_Vireo',
 'White_Necked_Raven',
 'White_Pelican',
 'White_Throated_Sparrow',
 'Wilson_Warbler',
 'Winter_Wren',
 'Worm_Eating_Warbler',
 'Yellow_Bellied_Flycatcher',
 'Yellow_Billed_Cuckoo',
 'Yellow_Breasted_Chat',
 'Yellow_Headed_Blackbird',
 'Yellow_Throated_Vireo',
 'Yellow_Warbler']
path = Path(__file__).parent

app = Starlette()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['X-Requested-With', 'Content-Type'])
app.mount('/static', StaticFiles(directory='app/static'))


async def download_file(url, dest):
    if dest.exists(): return
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.read()
            with open(dest, 'wb') as f:
                f.write(data)


async def setup_learner():
    await download_file(export_file_url, path / export_file_name)
    try:
        learn = load_learner(path, export_file_name)
        return learn
    except RuntimeError as e:
        if len(e.args) > 0 and 'CPU-only machine' in e.args[0]:
            print(e)
            message = "\n\nThis model was trained with an old version of fastai and will not work in a CPU environment.\n\nPlease update the fastai library in your training environment and export your model again.\n\nSee instructions for 'Returning to work' at https://course.fast.ai."
            raise RuntimeError(message)
        else:
            raise


loop = asyncio.get_event_loop()
tasks = [asyncio.ensure_future(setup_learner())]
learn = loop.run_until_complete(asyncio.gather(*tasks))[0]
loop.close()


@app.route('/')
async def homepage(request):
    html_file = path / 'view' / 'index.html'
    return HTMLResponse(html_file.open().read())


@app.route('/analyze', methods=['POST'])
async def analyze(request):
    img_data = await request.form()
    img_bytes = await (img_data['file'].read())
    img = open_image(BytesIO(img_bytes))
    prediction = learn.predict(img)[0]
    return JSONResponse({'result': str(prediction)})


if __name__ == '__main__':
    if 'serve' in sys.argv:
        uvicorn.run(app=app, host='0.0.0.0', port=5000, log_level="info")
