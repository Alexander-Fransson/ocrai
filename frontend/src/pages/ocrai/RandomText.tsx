import { useState, useRef, memo, FormEvent, ChangeEvent } from "react"

const characters: string = "<π!ŌŠ/нdêồfRkХÇỳñ©8•3yMôĠр>ņá(®ǒàơ¡J”èøěǔư≤.@î,тęşи âV¨s∂öằố[ừ6ḷaˇ7}ǐлpśòÖɔ±≠ºột{ů¯ﬂế\"KĐłćčÉÅçāğ†o^C14e∏¸žuh;ŞE˜=ü¥wēıểÄ°˙‚v2‹-ëуí9HČWœ€Yj:–Oụû&µк≥Q)0́äĀňAØ*̂m…xīPƒIú‘`Zšé|æăFïBṭō∞—óșãgńS£būßėả%5G‡´+ớz¬ʻq'⁄«˚rTḍ$“¿Lấ≈ªő~¢UiXÆð§N„›"

const createColour = () => {
    const red: number = Math.floor(Math.random() * 256)
    const green: number = Math.floor(Math.random() * 256)
    const blue: number = Math.floor(Math.random() * 256)
    
    return `rgb(${red },${green},${blue})`
}

const createRandomString = (fontsize: number) => {
    const spaceInCanvas = (700 * 1000)
    const spaceForChars =  spaceInCanvas / ((fontsize + 3)**2)
    const randomStringLength: number = Math.floor(Math.random() * spaceForChars  + 1)
    let randomString: string = ''

    for (let i = 0; i < randomStringLength; i++) {
        const randomCharacterLocation: number = Math.floor(Math.random() * (characters.length -2) + 1)
        randomString += characters[randomCharacterLocation]
    }
    return randomString
}

const choseRandomFont = () => {
    const fontList: FontFace[] = Array.from(document.fonts.keys())
    const fontNames: string[] = fontList.map(font => font.family)
    const randomIndex: number = Math.floor(Math.random() * fontNames.length)

    return fontNames[randomIndex]
}

const createImageData = (canvas: HTMLCanvasElement, isAnswer:boolean, textColour: string, backgroundColour: string, text:string, font: string, fontSize: number) => {
    const context = canvas.getContext('2d')
    const lineLength =  Math.floor(1000 / fontSize)
    const fitString = new RegExp(`.{1,${lineLength}}`, "g")
    const rows = text.match(fitString)
    const gap = Math.floor(Math.random() * 50 + 50)            
    let space = fontSize  

    // Clear canvas
    context!.clearRect(0, 0, canvas.width, canvas.height)

    // Fill backgroud
    if(isAnswer) backgroundColour = `rgb(255,255,255)`
    context!.fillStyle = backgroundColour
    context!.fillRect(0, 0, canvas.width, canvas.height);

    //Fill Text
    rows!.forEach((line:string) => {
        for (let i = 0; i < line.length; i++) {
            let char = line[i]
            const colourCode = characters.indexOf(char) + 1
            const x = gap + i * fontSize
            const y = gap + space 

            //Colours are ot pure. make boxes instead

            if(isAnswer) {
                textColour = `rgb(${colourCode},${colourCode},${colourCode})`
                context!.fillStyle = textColour
                context!.fillRect(x, y, fontSize - 5, fontSize -3)

                context!.strokeStyle = backgroundColour
                context!.strokeRect(x+0.5,y+0.5, fontSize - 5, fontSize -3)
            }else{
                context!.font = `${fontSize}px ${font}`
                context!.fillStyle = textColour
                context!.fillText(char, x, y)
            }
        }
        space += fontSize + 3
    })

    // Create Image File
    const dataUrl = canvas.toDataURL()
    const base64String = dataUrl.split(',')[1];
    const byteCharacters = atob(base64String);
    const byteArrays = [];
    for (let i = 0; i < byteCharacters.length; i++) {
        byteArrays.push(byteCharacters.charCodeAt(i));
    }
    const byteArray = new Uint8Array(byteArrays);
    
    return new File([byteArray], 'image.png', { type: 'image/png' });
}

const RandomText = memo(() => {
    let fontSize =  Math.floor(Math.random() * 10 + 10)
    let backgroundColour = createColour()
    let textColour = createColour()
    let randomText = createRandomString(fontSize)
    let randomFont = choseRandomFont()
    const [imgSrc, setImgSrc] = useState('')
    let epochs = 1

    const questionData = useRef<HTMLCanvasElement | null>(null)
    const answerData = useRef<HTMLCanvasElement | null>(null) 

    const change = () => {
        fontSize = Math.floor(Math.random() * 10 + 10)
        randomFont = choseRandomFont()
        backgroundColour = createColour()
        textColour = createColour()
        randomText = createRandomString(fontSize)
    }

    const makePrediction = (event?: FormEvent) => {
        if(event) event.preventDefault()

        const question = questionData.current
        const answer = answerData.current

        if(question && answer){
            const questionImage = createImageData(question, false, textColour, backgroundColour, randomText, randomFont, fontSize)
            createImageData(answer, true, textColour, backgroundColour, randomText, randomFont, fontSize)

            const bodyData = new FormData()
            bodyData.append('image', questionImage)

            fetch('http://127.0.0.1:8000/ocrai/predict', {
                method: 'POST',
                body: bodyData
            })
            .then(data => data.json())
            .then(json => setImgSrc(json.url))
            .catch(error => {
                throw new Error(error)
            })

            change()
        }
    }

    const trainModel = (event?: FormEvent) => {
        if(event) event.preventDefault()

        const question = questionData.current
        const answer = answerData.current

        if(question && answer){
            const questionImage = createImageData(question, false, textColour, backgroundColour, randomText, randomFont, fontSize)
            const answerImage = createImageData(answer, true, textColour, backgroundColour, randomText, randomFont, fontSize)

            const bodyData = new FormData()
            bodyData.append('image', questionImage)
            bodyData.append('answer_as_pixles', answerImage)

            change()

            fetch('http://127.0.0.1:8000/ocrai', {
                method: 'POST',
                body: bodyData
            })
            .then(data => data.json())
            .then(res => {
                if(epochs > 0){
                    epochs --
                    console.log(epochs)
                    trainModel()
                }else{
                    console.log(res)
                    makePrediction()
                }
            })

        }
    }
    
    const handleEpochChange = (event: ChangeEvent<HTMLInputElement>) => {
        let value: number = parseInt(event.target.value)

        if(isNaN(value)){
            value = 1
        }

        epochs = value
        
    }

    //Bådat talen måste kunna delas med två fyra gånger
    
    return (
        <>

            <h5>make prediction</h5>
            <form onSubmit={makePrediction}>
                <input type="submit"/>
            </form>

            <h5>train Model</h5>
            <form onSubmit={trainModel}>
                <input type='number' placeholder="epochs" min={1} onChange={handleEpochChange} id="" />
                <input type="submit" />
            </form>
            <img src={imgSrc} alt="" />
            <canvas ref={questionData} width='1120' height='800'/>
            <canvas ref={answerData} width='1120' height='800' />
        </>
    );
})

export default RandomText