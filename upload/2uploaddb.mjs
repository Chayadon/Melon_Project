import { initializeApp } from "firebase/app";
import { getFirestore, collection, query, Timestamp,setDoc,doc} from "firebase/firestore";
import { getStorage, ref, getDownloadURL } from "firebase/storage";
import { readFileSync } from "fs";

const firebaseConfig = {
    apiKey: "AIzaSyAhfpKgLgE553HnqUjVgQeTNrGmYaZMShU",
    authDomain: "web-melon01.firebaseapp.com",
    projectId: "web-melon01",
    storageBucket: "web-melon01.appspot.com",
    messagingSenderId: "437275991929",
    appId: "1:437275991929:web:7af5b4e5b0f0745a8499a8",
    measurementId: "G-RPG07G5C8X"
};


const app = initializeApp(firebaseConfig);
const db = getFirestore();
const q = query(collection(db, "Melon"));



function syncReadFile(filename) {
    const contents = readFileSync(filename, 'utf-8');

    const arr = contents.split(/\r?\n/);

    return arr;

}

const Confi = syncReadFile('./Confidence_value.txt');
const Bound = syncReadFile('./Bounding_box.txt');
const pathimg = readFileSync('./pathimg.txt', 'utf-8');;

console.log(pathimg);

const storage = getStorage();
const starsRef = ref(storage, pathimg);
let id = pathimg.toString().split("-");

id.shift();
id.shift();
id = id.toString().split(".")
id.pop()
getDownloadURL(starsRef)
    .then(async (url) => {

        const docData = {
            Camera: "cam1",
            Confidence_value: Confi.toString(),
            Bounding_box: Bound.toString(),
            URL: url.toString(),
            Date: Timestamp.fromDate(new Date()),
        };

        console.log(docData);

        await setDoc(doc(db, "Melon", "cam1-" + id.toString() ), docData);


    });

    //uplode to firestroage









