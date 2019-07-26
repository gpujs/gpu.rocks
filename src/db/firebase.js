import * as firebase from 'firebase/app';
import * as admin from 'firebase-admin';

const firebaseConfig = {
  apiKey: "AIzaSyDqLgu5ijPw4njzJTyzmLAIYl9a3DVpfiI",
  authDomain: "gpujs-9b507.firebaseapp.com",
  databaseURL: "https://gpujs-9b507.firebaseio.com",
  projectId: "gpujs-9b507",
  storageBucket: "",
  messagingSenderId: "675667047799",
  appId: "1:675667047799:web:656312850eb7acbf"
}

firebase.initializeApp(firebaseConfig);

admin.initializeApp({
  credential: admin.credential.applicationDefault()
})

const getDb = () => admin.firestore();

export default getDb;