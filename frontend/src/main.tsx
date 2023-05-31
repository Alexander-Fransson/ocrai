import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './pages/app/App.tsx'
import RandomText from './pages/ocrai/RandomText.tsx'
import './index.css'
import { createBrowserRouter, RouterProvider} from 'react-router-dom'

const router = createBrowserRouter([
  {path:"/", element:<App />},
  {path:"/ocrai", element:<RandomText/>}
])

ReactDOM.createRoot(document.getElementById('root') as HTMLElement).render(
  <React.StrictMode>
    <RouterProvider router={router}/>
  </React.StrictMode>,
)
