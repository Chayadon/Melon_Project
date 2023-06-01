import { initializeApp } from "firebase/app";
import { getFirestore, collection, query} from "firebase/firestore";
import { getStorage, ref, getDownloadURL } from "firebase/storage";
import { readFileSync, rename, readdirSync ,writeFile} from "fs";
import path from 'path';
import { fileURLToPath } from 'url';
import admin from 'firebase-admin';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

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

let i = 1;
const mili = +new Date();




function getCurrentFilenames() {
    readdirSync(__dirname).forEach(file => {


         if(i == 4){

              rename(file, mili + '.jpg', (error) => {
                 if (error) {

                      console.log(error);

                  }
                  else {

                      console.log("\nFile Renamed\n");


                  }
              });
         console.log(file);

         }
        i++;

    });


}


var serviceAccount = "./web-melon01-firebase-adminsdk-67r3m-fcae7459c8.json";
admin.initializeApp({
    credential: admin.credential.cert(serviceAccount),
    storageBucket: "web-melon01.appspot.com"
});
var bucket = admin.storage().bucket();

var filename = "./" + mili + ".jpg";
//var filename = "./123.jpg";

export const destinationFilename = 'Melon-img/cam1-' + mili + '.jpg';

async function uploadFile() {
    await bucket.upload(filename, {
        gzip: true,
        destination: destinationFilename
    });
    console.log(`${filename} uploaded to ` + destinationFilename);
}
//upload to Storage


let data = destinationFilename;

// Write data in 'Output.txt' .
writeFile('./pathimg.txt', data, (err) => {
    // In case of a error throw err.
    if (err) throw err;
});

getCurrentFilenames();
uploadFile();
